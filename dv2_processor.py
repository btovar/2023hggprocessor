from coffea.nanoevents import NanoEventsFactory, BaseSchema, PFNanoAODSchema
import json
import fastjet
import numpy as np
import awkward as ak
from coffea import processor
import hist
import coffea.nanoevents.methods.vector as vector
import warnings
import hist.dask as dhist
import dask
import pickle
import os
import distributed
from ndcctools.taskvine import DaskVine
import time

full_start = time.time()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "The necessary")

    dataset_files = [
        "Hgg",
        "Hbb",
        "QCD_Pt_300to470",
        "QCD_Pt_470to600",
        "QCD_Pt_600to800",
        "QCD_Pt_800to1000",
        "QCD_Pt_1000to1400",
        "QCD_Pt_1400to1800",
        "QCD_Pt_1800to2400",
        "QCD_Pt_2400to3200",
        "QCD_Pt_3200toInf",
    ]

    source = "/project01/ndcms/cmoore24"

    events = {}
    for name in dataset_files:
        with open(f"filelists/{name}") as f:
            events[name] = NanoEventsFactory.from_root(
                [{f"{source}/{fn.strip()}": "/Events"} for fn in f.readlines()],
                #permit_dask=True,
                schemaclass=PFNanoAODSchema,
                metadata={"dataset": name},
            ).events()

    def color_ring(fatjet):
        # return color_ring_dv2(fatjet)
        return color_ring_sp(fatjet)

    def color_ring_dv2(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.exclusive_subjets_up_to(
            data=cluster.exclusive_jets(n_jets=1), nsub=3
        )  # uncomment this when using C/A
        # subjets = cluster.inclusive_jets()
        vec = ak.zip(
            {
                "x": subjets.px,
                "y": subjets.py,
                "z": subjets.pz,
                "t": subjets.E,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        i, j = ak.unzip(ak.combinations(vec, 2))
        best = ak.argmax((i + j).mass, axis=1, keepdims=True)
        leg1, leg2 = ak.firsts(i[best]), ak.firsts(j[best])
        # assert ak.all((leg1 + leg2).mass == ak.max((i + j).mass, axis=1))
        # leg3 = vec[(best == 0)*2 + (best == 1)*1 + (best == 2)*0]
        leg3 = ak.firsts(vec[(vec.idx != leg1.idx) & (vec.idx != leg2.idx)])
        # assert ak.all(leg3.x != leg1.x)
        # assert ak.all(leg3.x != leg2.x)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        color_ring = (a13**2 + a23**2) / (a12**2)
        return color_ring

    def color_ring_sp(fatjet, variant=False):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.2)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.inclusive_jets()
        vec = ak.zip(
            {
                "x": subjets.px,
                "y": subjets.py,
                "z": subjets.pz,
                "t": subjets.E,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        i, j, k = ak.unzip(ak.combinations(vec, 3))
        best = ak.argmin(abs((i + j + k).mass - 125), axis=1, keepdims=True)
        order_check = ak.concatenate([i[best].mass, j[best].mass, k[best].mass], axis=1)
        largest = ak.argmax(order_check, axis=1, keepdims=True)
        smallest = ak.argmin(order_check, axis=1, keepdims=True)
        leading_particles = ak.concatenate([i[best], j[best], k[best]], axis=1)
        leg1 = leading_particles[largest]
        leg3 = leading_particles[smallest]
        leg2 = leading_particles[
            (leading_particles.idx != ak.flatten(leg1.idx))
            & (leading_particles.idx != ak.flatten(leg3.idx))
        ]
        leg1 = ak.firsts(leg1)
        leg2 = ak.firsts(leg2)
        leg3 = ak.firsts(leg3)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        if variant is False:
            color_ring = (a13**2 + a23**2) / (a12**2)
        else:
            color_ring = a13**2 + a23**2 - a12**2
        return color_ring

    def d2_calc(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )  # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        d2 = softdrop_cluster.exclusive_jets_energy_correlator(func="D2")
        return d2

    class MyProcessor_Signal(processor.ProcessorABC):
        def __init__(self):
            pass

        def process(self, events):
            dataset = events.metadata["dataset"]

            fatjet = events.FatJet

            genhiggs = events.GenPart[
                (events.GenPart.pdgId == 25)
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
            parents = events.FatJet.nearest(genhiggs, threshold=0.1)
            higgs_jets = ~ak.is_none(parents, axis=1)

            cut = (
                (
                    (fatjet.pt > 300)
                    & (fatjet.msoftdrop > 110)
                    & (fatjet.msoftdrop < 140)
                    & (abs(fatjet.eta) < 2.5)
                )
                & (higgs_jets)
                & (fatjet.btagDDBvLV2 > 0.65)
            )

            #boosted_fatjet = ak.mask(fatjet, cut)
            boosted_fatjet = fatjet[cut]

            uf_cr = ak.unflatten(
                color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet)
            )
            d2 = ak.unflatten(d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            boosted_fatjet["color_ring"] = uf_cr
            boosted_fatjet["d2b1"] = d2

            hcr = dhist.Hist.new.Reg(
                40, 0, 10, name="color_ring", label="Color_Ring"
            ).Weight()

            d2b1 = dhist.Hist.new.Reg(40, 0, 3, name="D2B1", label="D2B1").Weight()

            cmssw_n2 = dhist.Hist.new.Reg(
                40, 0, 0.5, name="cmssw_n2", label="CMSSW_N2"
            ).Weight()

            cmssw_n3 = dhist.Hist.new.Reg(
                40, 0, 3, name="cmssw_n3", label="CMSSW_N3"
            ).Weight()

            ncons = dhist.Hist.new.Reg(
                40, 0, 200, name="constituents", label="nConstituents"
            ).Weight()

            mass = dhist.Hist.new.Reg(40, 0, 250, name="mass", label="Mass").Weight()

            sdmass = dhist.Hist.new.Reg(
                40, 0, 250, name="sdmass", label="SDmass"
            ).Weight()

            btag = dhist.Hist.new.Reg(40, 0, 1, name="Btag", label="Btag").Weight()

            fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            hcr.fill(color_ring=fill_cr)
            d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
            cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
            cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
            ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
            mass.fill(mass=ak.flatten(boosted_fatjet.mass))
            sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
            btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))

            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    "Color_Ring": hcr,
                    "N2": cmssw_n2,
                    "N3": cmssw_n3,
                    "nConstituents": ncons,
                    "Mass": mass,
                    "SDmass": sdmass,
                    "Btag": btag,
                    "D2": d2b1,
                }
            }

        def postprocess(self, accumulator):
            pass

    class MyProcessor_Background(processor.ProcessorABC):
        def __init__(self):
            pass

        def process(self, events):
            dataset = events.metadata["dataset"]

            fatjet = events.FatJet

            cut = (
                (fatjet.pt > 300)
                & (fatjet.msoftdrop > 110)
                & (fatjet.msoftdrop < 140)
                & (abs(fatjet.eta) < 2.5)
            ) & (fatjet.btagDDBvLV2 > 0.65)

            boosted_fatjet = fatjet[cut]

            uf_cr = ak.unflatten(
                color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet)
            )
            d2 = ak.unflatten(d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            boosted_fatjet["color_ring"] = uf_cr
            boosted_fatjet["d2b1"] = d2

            hcr = dhist.Hist.new.Reg(
                40, 0, 10, name="color_ring", label="Color_Ring"
            ).Weight()

            d2b1 = dhist.Hist.new.Reg(40, 0, 3, name="D2B1", label="D2B1").Weight()

            cmssw_n2 = dhist.Hist.new.Reg(
                40, 0, 0.5, name="cmssw_n2", label="CMSSW_N2"
            ).Weight()

            cmssw_n3 = dhist.Hist.new.Reg(
                40, 0, 3, name="cmssw_n3", label="CMSSW_N3"
            ).Weight()

            ncons = dhist.Hist.new.Reg(
                40, 0, 200, name="constituents", label="nConstituents"
            ).Weight()

            mass = dhist.Hist.new.Reg(40, 0, 250, name="mass", label="Mass").Weight()

            sdmass = dhist.Hist.new.Reg(
                40, 0, 250, name="sdmass", label="SDmass"
            ).Weight()

            btag = dhist.Hist.new.Reg(40, 0, 1, name="Btag", label="Btag").Weight()

            fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            hcr.fill(color_ring=fill_cr)
            d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
            cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
            cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
            ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
            mass.fill(mass=ak.flatten(boosted_fatjet.mass))
            sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
            btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))

            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    "Color_Ring": hcr,
                    "N2": cmssw_n2,
                    "N3": cmssw_n3,
                    "nConstituents": ncons,
                    "Mass": mass,
                    "SDmass": sdmass,
                    "Btag": btag,
                    "D2": d2b1,
                }
            }

        def postprocess(self, accumulator):
            pass

    start = time.time()
    result = {}
    result["Hgg"] = MyProcessor_Signal().process(events["Hgg"])
    print("hbb")
    result["Hbb"] = MyProcessor_Signal().process(events["Hbb"])
    print("300")
    result["QCD_Pt_300to470_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["QCD_Pt_300to470"]
    )
    print("470")
    result["QCD_Pt_470to600_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["QCD_Pt_470to600"]
    )
    print("600")
    result["QCD_Pt_600to800_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["QCD_Pt_600to800"]
    )
    print("800")
    result["QCD_Pt_800to1000_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["QCD_Pt_800to1000"]
    )
    print("1000")
    result[
        "QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["QCD_Pt_1000to1400"])
    print("1400")
    result[
        "QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["QCD_Pt_1400to1800"])
    print("1800")
    result[
        "QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["QCD_Pt_1800to2400"])
    print("2400")
    result[
        "QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8"
    ] = MyProcessor_Background().process(events["QCD_Pt_2400to3200"])
    print("3200")
    result["QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8"] = MyProcessor_Background().process(
        events["QCD_Pt_3200toInf"]
    )
    stop = time.time()
    print(stop - start)

    print("computing")
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-color-ring",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    computed = dask.compute(
        result,
        scheduler=m,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=False,
    )
    with open("big_btagged_result.pkl", "wb") as b:
        pickle.dump(computed, b)


full_stop = time.time()
print("full run time is " + str((full_stop - full_start) / 60))
