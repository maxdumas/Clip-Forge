# Text2Building

## Introduction

This repository contains the code for Text2Building, an attempt by myself to
adapt the code for the paper "CLIP-Forge: Towards Zero Shot Text-to-Shape
Generation" to the domain of 3D building models.

The goal of this project is to explore using cutting-edge breakthroughs in
generative artficial intellience to democratize the building design process. 

Buildings and cities have become dramatically more complex over the last 150
years. The 2022 revision of the [New York City construction
codes](https://www1.nyc.gov/site/buildings/codes/2022-construction-codes.page),
which just came into effect, has 88 chapters and 34 appendices. 15 chapters
alone refer to plumbing codes. Mandates for increased sustainability add further
complexity. The creation of buildings has [never been more
expensive](https://www.turnerconstruction.com/cost-index), and yet cities need
new construction to become more sustainable and equitable. In the United States
as a whole, the Federal Housing Finance Agency house price index shows that an
average home costs [5.5 times what it did in
1980](https://fred.stlouisfed.org/series/USSTHPI). This is causing a true
affordability crisis.

In tandem with physical complexity, the socio-political problems around
construction within our urban centers have also risen: Some communities have
grown increasingly distrustful of the development process due to a long history
of
[misguided](https://www.urbandisplacement.org/about/what-are-gentrification-and-displacement/)
or
[malevolent](https://www.nytimes.com/2021/08/17/realestate/what-is-redlining.html)
urban development practices. Other communities have economic incentives that are
unaligned with the greater good, and want constricted housing supply so as to
increase their own home prices. Often these groups are wealthy and wield
disproportionate political power. This persists despite a growing body of
research showing that the lack of affordable housing drives many other social
maladies, like rising levels of obesity, falling fertility rates, low
productivity growth, and rising wealth inequality.

Satisfactory answers to this crisis are elusive. Do we demolish historic
neighborhoods to replace them with highrises? Do we sacrifice on sustainable
building practices to make homes cheaper? Do we relax building codes, perhaps
reducing the resiliency and even safety of our new buildings? The answers are
impossible to arrive at axiomatically, and will depend on each community and
situation. This set of indeterminate contrasting forces obstructing any clear
solution to our predicament is known as a Wicked Problem. In the face of Wicked
Problems, an effective solution is iterative design incorporating extensive
feedback from all affected parties. Stakeholders present their ideal visions to
one another and see where the gaps between their visions lie, and then create a
new shared vision that attempts to find a middle ground between their competing
ideals. This process does not succeed after the first attempt, and needs to be
repeated until [sufficient
trust](https://www.nytimes.com/2021/12/02/us/hurricane-sandy-lower-manhattan-nyc.html)
is established and until enough dimensions of the contrasting visions have been
hashed out and resolved. This process is slow, expensive, and time-consuming,
but necessary and important.

Slow, expensive processes are often the first to be cut under time and budgetary
constraints. A clear way that tech can help is by making the process of
designing buildings faster and cheaper. Architects typically charge [around
10%](https://www.architectmagazine.com/practice/an-architects-fees-whats-your-time-worth_o)
of the construction cost for a project, and closer to 12% in NYC. Turnaround
times can be days, weeks, or months, depending on the complexity of the project.
Reducing these costs would allow for cheaper, faster iteration and thus more
iteration. The more attempts stakeholders can have to come to a shared vision,
the more satisfactory the end result is likely to be for all parties.

The hope is that generative algorithms, especially generative AI, can be
leveraged to dramatically reduce these design costs and speed up iteration
cycles. My hope is that these techniques will also help to democratize the
design conversation: If going back to the drawing board after receiving critical
feedback during a community meeting costs only a modicum of time and money, it
may become more possible to deeply consider the needs of all people when
building while still reducing costs.

## Literature Review

The field of 3D generative AI is evolving extremely rapidly. Extensive work is
being done to enable the creation of 3D geometry with a variety of different
input modalities, such as 2D images, incomplete 3D geometry, point clouds,
videos, and text.

[TODO]

## Dataset Creation
Considered the following datasets:
* Berlin 3D https://www.businesslocationcenter.de/en/economic-atlas/download-portal/
* Montreal 3D https://donnees.montreal.ca/dataset/maquette-numerique-plateau-mont-royal-batiments-lod2-avec-textures
* NYC 3D https://github.com/CityOfNewYork/nyc-geo-metadata/blob/master/Metadata/Metadata_3DBuildingModel.md
In the end, settled on BuildingNet: https://buildingnet.org/

I considered several different possible datasets when starting this project. My
first thought was to use one or more of the existing 3D building model datasets
that exist for real-life cities. Such datasets exist for a number of cities,
such as [New York
City](https://github.com/CityOfNewYork/nyc-geo-metadata/blob/master/Metadata/Metadata_3DBuildingModel.md),
[Boston](https://www.bostonplans.org/3d-data-maps/3d-smart-model/about-3d),
[Montreal](https://donnees.montreal.ca/ville-de-montreal/maquette-numerique-plateau-mont-royal-batiments-lod2-avec-textures),
and
[Berlin](https://www.businesslocationcenter.de/en/economic-atlas/download-portal/).
These datasets are compelling because of their vastness. For example, the NYC
datatset for Manhattan alone contains 45,848 3D building models. They are also
compelling because, being derived from real buildings, there is no risk of a
domain gap between our training data and my goal, which is to generate realistic
building models from text.

Ultimately, these datasets proved unsatisfactory, however. These building models
mostly contain 3D models in "LOD 1", meaning that only slightly more detail than
a bounding box for the building is captured. Some of these datasets capture more
roof details, but very few capture geometric facade details. These datasets are
generally intended for low-detail visualization or macro-scale simulation and
analysis, and are not well-suited to architectural visualization.

Another dataset, [BuildingNet](https://buildingnet.org/), ended up being better
aligned to this task. This dataset contains 2,000 human-created, detailed 3D models
of buildings. These models are largely of the style and form that one would
expect for architectural visualization—most seem to have been made in SketchUp
or similar tools.

These 2,000 3D models require extensive normalization and are derived into a
series of other artifacts. The code for this is in [the sibling
repo](https://github.com/maxdumas/text2building_data) for this project.

Normalization steps include:
* Turning each mesh into a watertight mesh using [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus).
* Removing isolated dangling components from each mesh. This is a heuristic to
  remove components from the meshes that may not correspond to the main building
  mass, such as trees, people, cars, etc.
* Removing the ground plane from each mesh. A number of meshes include some sort
  of floor surface which in some cases is significantly larger than the building
  mesh itself.

Artifact generation steps included:
* Rendering textured, shaded images of each building mesh from 12 different
  evenly spaced angles rotated around the Z-axis.
* Generating 32x32x32 BINVOX voxel representations of each building mesh using
  [`cuda_voxelizer`](https://github.com/maxdumas/cuda_voxelizer).
* Generating 100,000-element occupancy point clouds by sampling points at
  randomly within the bounding box of the voxel volume and checking if they were
  contained by the voxel volume itself.

## Training Infrastructure 
* PyTorch Lightning
* SageMaker Spot Training
* Weights & Biases

## Model Architecture

The model architecture largely mirrors the structure of the CLIP-Forge paper.
Training occurs in two phases: In the first phase, I train an autoencoder to
learn a lower-dimensional representation of 32x32x32 voxel representations of
the 3D building meshes. In the second phase, I generate 2D image renderings of
the 3D building meshes from a variety of angles and compute CLIP embeddings
using those images. I then train a Latent Flows network to map the distribution
of the learned autoencoder embeddings to the distribution of the CLIP
embeddings.

During inference, the input is a text phrase. This text phrase is encoded into
an embedding by CLIP, which can then be brought into the embedding space of the
autoencoder using the Latent Flows network. We then decode the embedding to
arrive at a voxel representation of the text phrase.

The autoencoder differs somewhat from the CLIP-Forge autoencoder in that it is a
variational autoencoder (VAE), whose learned features attempt to model a
high-dimensional gaussian representation of the principal features of the
training data, as opposed to trying to encode the features directly. This type
of autoencoder proved to be more robust in my testing, likely due to a VAE's
ability to generate plausible results when randomly sampled. The final
autoencoder consists of 5 layers of batch-normalized 3D convolutional layers
with rectified linear unit activiation and a final fully-connected layer.

The Latent Flows network remains identical to that used in the CLIP-Forge paper.
This network consists of a sequence of 5 batch-normalized "coupling layers",
which in turn each contain scale and translate networks. The scale networks
consist of 3 fully connected layers interspersed with tanh activation, and the
translation networks consist of 3 fully connected layers interspersed with
rectified linear unit activation.

[CLIP](https://openai.com/research/clip) is a model provided by OpenAI and is
considered frozen for the purposes of this experiment. Its use, as can be
inferred from the above, is to join the text and 2D image modalities into a single
shared embedding space. I am interested in trying out other competing models for
generating joint text-image embeddings in the future.

## Results

## Next Steps

### Additional Data
* RealCity3D
LOD-2 3D models of every building in NYC and Zurich, made easily usable. In particular, the Zurich buildings seem interesting.

* Blackshark.ai
* Cities Skylines dataset
* Objaverse

Recently released by the Allen Institute for AI, the [Objaverse
dataset](https://objaverse.allenai.org/) consists of over 800,000 annotated 3D
models sourced from throughout the Internet. Preliminiary investigation shows
that this dataset includes at least 1,000 models of buildings that are easily
discoverable. The potential of this dataset is significant, as it could easily
double or more the usable data for this experiment. They key issue with this
data is that the included data varies widely in quality, detail, style, and
technique. Some building models are in a cartoon style intended for animation,
others for videogames. Some models may be highly normalized and intended for
usage in CAD software, others may be constructed by photogrammetry of real
buildings and as such have extremely noisy meshes. Figuring out how to properly
normalize these meshes will be key to using them in this project.

* Neural rendering techniques, fantasia, AutoSDF. Fine-tuning emerged foundation models.

* Use CLIP alternatives

[CLIP](https://openai.com/research/clip) is a model provided by OpenAI and is
considered frozen for the purposes of this experiment. Its use, as can be
inferred from the above, is to join the text and 2D image modalities into a single
shared embedding space. I am interested in trying out other competing models for
generating joint text-image embeddings in the future.

* Dive deeper into Latent Flows network architecure