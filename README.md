# ML-models
Collection of various ML models developed for the MaMMoS project

Models are organized in different folders of this repository:


- ./beyond-stoner-wolfarth:

   ML models to replace single-grain micromagnetics simulations by ML predictions for:

   -- ./beyond-stoner-wolfarth/single-grain-easy-axis-model:
      where the H and K1 vector are aligned along the z-axis of the simulation cube
      This model predicts Mr, Hc, and BHmax from a, k1, and Ms
      Current model version: v1.0

   -- ./inverse-beyond-stoner-wolfarth/single-grain-easy-axis-model:
      where the H and K1 vector are aligned along the z-axis of the simulation cube
      This inverse model predicts a, k1, and Ms from Mr, Hc, and BHmax
      Current model version: not yet publises

   -- ./beyond-stoner-wolfarth/single-grain-mutli-angle-model:
      where the H and K1 vector can independently point in any direction
      This models predicts Mr, Hc, and BHmax from a, k1, and Ms
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

