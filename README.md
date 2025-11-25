# ML-models
Collection of various ML models for the MaMMoS project

Models are organized in different folders of this repository:


- ./beyond-stoner-wolfarth:

   ML models to replace single-grain micromaggnetics simulations by ML predictions for

   -- ./beyond-stoner-wolfarth/single-grain-easy-axis-model:
      where the H and K1 vector are aligned along the z-axis of the simulation cube


   -- ./beyond-stoner-wolfarth/single-grain-mutli-angle-model:
      where the H and K1 vector can independently point in any direction
      
- ./experimental-simulation-ms:

    Identify and correct systematic biases in DFT-based simulations of spontaneous magnetization, using statistical modeling of error patterns,
    AI-driven correction functions, data augmentation for missing experimental values (⚠️ Not yet validated and may introduce bias).


