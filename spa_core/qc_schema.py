from __future__ import annotations


QC_AUDIT_SCHEMA = [
    {
        "artifact_family": "Additive interference",
        "physical_phenomenon": "External signals superimposed on speech",
        "gui_name": "Environmental noise",
        "effects": [
            "Traffic",
            "HVAC",
            "Pets",
            "TV (non-speech)",
            "Beep",
            "Microphone rubbing",
            "Non-specific environmental noise",
        ],
        "potential_human_qc_linking": "Background noise",
    },
    {
        "artifact_family": "Temporal discontinuities",
        "physical_phenomenon": "Disruptions in time structure of the signal",
        "gui_name": "Any non-task related content",
        "effects": [
            "Extra or filler word",
            "Missed word",
            "Lip smacking/mouth sounds",
            "Mouse clicking/keyboard noise",
            "Laughing",
            "Coughing",
        ],
        "potential_human_qc_linking": "NA",
    },
    {
        "artifact_family": "Additive interference",
        "physical_phenomenon": "External signals superimposed on speech",
        "gui_name": "Competing speech",
        "effects": ["TV (speech)", "Other human speakers"],
        "potential_human_qc_linking": "Another person speaks",
    },
    {
        "artifact_family": "Gain / level dynamics",
        "physical_phenomenon": "Time-varying amplitude scaling of the signal",
        "gui_name": "Volume unstable",
        "effects": ["Volume too quiet", "Volume too loud", "Volume changes"],
        "potential_human_qc_linking": "Volume unstable",
    },
    {
        "artifact_family": "Nonlinear distortion",
        "physical_phenomenon": "Amplitude-dependent signal deformation",
        "gui_name": "Clipping",
        "effects": [
            "Crackling on loud syllables",
            "Grainy, sandy, buzzing voice texture",
            "Whole voice sounds crushed or overloaded",
        ],
        "potential_human_qc_linking": "Volume unstable",
    },
    {
        "artifact_family": "Reverberation / echo",
        "physical_phenomenon": "Acoustic reflections in the environment",
        "gui_name": "Reverberation/echo",
        "effects": ["Reverb", "Echo"],
        "potential_human_qc_linking": "Poor audio quality",
    },
    {
        "artifact_family": "Channel / device / platform",
        "physical_phenomenon": "Signal transformation by hardware and software pipeline",
        "gui_name": "Platform effects",
        "effects": [
            "Muffled, filtered, telephone-like",
            "Underwater, robotic, warbly, codec-like",
        ],
        "potential_human_qc_linking": "Poor audio quality",
    },
    {
        "artifact_family": "Temporal discontinuities",
        "physical_phenomenon": "Disruptions in time structure of the signal",
        "gui_name": "Temporal discontinuities",
        "effects": ["Audio lagging", "Audio glitching", "Audio skipped/missed"],
        "potential_human_qc_linking": "NA",
    },
]

QC_AUDIT_GROUPS = [(item["gui_name"], item["effects"]) for item in QC_AUDIT_SCHEMA]
QC_AUDIT_GUI_NAMES = [item["gui_name"] for item in QC_AUDIT_SCHEMA]
QC_AUDIT_METADATA = {
    (item["gui_name"], effect): {
        "artifact_family": item["artifact_family"],
        "physical_phenomenon": item["physical_phenomenon"],
        "potential_human_qc_linking": item["potential_human_qc_linking"],
    }
    for item in QC_AUDIT_SCHEMA
    for effect in item["effects"]
}
