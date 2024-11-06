# BCon
The repository for utilizing ControlNet to generate more realistic and diverse synthetic images.

## Guide for Installation and Setting Up Environment

### Using the Conda Environment YAML File
If you prefer to create the environment using the provided environment.yaml file, you can follow these steps:

1. Create the Environment from the YAML File:

```sh
   conda env create -f environment.yaml

2. Activate the Environment:

```sh
   conda activate bcon



### Setting Up a New Environment

1. **Create a new Conda environment** (recommended):
   ```sh
   conda create --name bcon python=3.9.20
   
2. **Activate the environment:**

   ```sh
   conda activate bcon

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
