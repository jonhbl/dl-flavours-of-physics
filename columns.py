features = [
    # Tau parameters
    "LifeTime",  # Life time of tau candidate.
    "dira",  # Cosine of the angle between the τ momentum and line between PV and tau vertex.
    "pt",  # Transverse momentum of τ
    "FlightDistance",  # Distance between τ and primary vertex (PV)
    "FlightDistanceError",
    "VertexChi2",
    "IP",  # Impact Parameter of tau candidate.
    "IPSig",  # Significance of Impact Parameter
    # Muons parameters
    ## Impact
    "p0_IP",
    "p1_IP",
    "p2_IP",
    "p0_IPSig",
    "p1_IPSig",
    "p2_IPSig",
    "IP_p0p2",  # Impact parameter of the p0 and p2 pair.
    "IP_p1p2",
    ## Kinematics
    "p0_pt",  # Transverse momentum
    "p1_pt",
    "p2_pt",
    "p0_p",  # Momentum
    "p1_p",
    "p2_p",
    "p0_eta",  # Pseudorapidity
    "p1_eta",
    "p2_eta",
    "DOCAone",  # Distance of Closest Approach between p0 and p1
    "DOCAthree",
    "DOCAtwo",
    "SPDhits",  # Hits in the SPD (Spin Physics Detector)
    ## Track
    "iso",  # Track isolation variable.
    "isolationa",
    "isolationb",
    "isolationc",
    "isolationd",
    "isolatione",
    "isolationf",
    "ISO_SumBDT",
    "p0_IsoBDT",
    "p1_IsoBDT",
    "p2_IsoBDT",
    "p0_track_Chi2Dof",  #  Quality of p0 muon track.
    "p1_track_Chi2Dof",
    "p2_track_Chi2Dof",
    ## Cone
    "CDF1",  # Cone isolation variable
    "CDF2",
    "CDF3",
]

extra_params = [
    "production",
    "signal",
    "id",
    "mass",
    "min_ANNmuon",
]
