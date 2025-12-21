# ML-models
Collection of various ML models for the MaMMoS project

Models are organized in different folders of this repository:


- ./beyond-stoner-wolfarth:

   ML models to replace single-grain micromaggnetics simulations by ML predictions for

   -- ./beyond-stoner-wolfarth/single-grain-easy-axis-model:
      where the H and K1 vector are aligned along the z-axis of the simulation cube
      Current model version: v0.1


   -- ./beyond-stoner-wolfarth/single-grain-mutli-angle-model:
      where the H and K1 vector can independently point in any direction
      Current model version: v0.1
      
- ./experimental-simulation-hc:
    Identify and correct systematic errors between micromagnetics simulations and experimental measurements of Hc
      Current model version: v0.1

- ./experimental-simulation-ms:

    Identify and correct systematic errors between simulations of the spontaneous magnetization and experimental measurement
    (⚠️ Not yet validated and may introduce bias).

      Current model version: v0.1

- ./experimental-simulation-tc:

    Identify and correct systematic errors between simulations of the Curie temperature and experimental measurement
    (⚠️ Not yet validated and may introduce bias).

      Current model version: v0.1

