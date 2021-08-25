import os
import re
import requests
from typing import List

import pymatgen.ext.matproj

def get_parent_structure_id(id_2dm: str) -> str:
    '''
    Given a 2DMatPedia ID, queries the 2DMatPedia for the structure's parent bulk's MaterialsProject ID
    If the material was a bottom-up material, or no parent is found, will return the string "no_parent" instead.

    Notes:
        The 2D Materials Encyclopedia can be found at http://www.2dmatpedia.org/ as of 8/25/21.

    Args:
        id_2dm (str): The ID of the material on the 2D Materials encyclopedia
    '''
    url = f"http://www.2dmatpedia.org/2dmaterials/doc/{id_2dm}"
    request = requests.get(url)

    mp_results = list(filter(lambda line: re.search("Obtained from 3D", line),
                             request.text.split("\n")
                             )
                      )

    if len(mp_results) == 0:
        result = "no_parent"
    else:
        result = re.search("mp-\d+", mp_results[0])[0]
    return result

def get_e_above_hull(material_id: str, pymatgen_rester: pymatgen.ext.matproj.MPRester) -> float:
    '''
    Looks up the energy above hull within Materials Project.

    Notes:
        - The Materials Project can be found at https://materialsproject.org/ as of 8/25/21.
        - Energy above hull is defined on Materials Project in the following glossary entry:
            https://docs.materialsproject.org/user-guide/glossary-of-terms/#energetics
        - The text of the glossary entry is reproduced below (in case the link goes bad), and is current as of 8/25/21:
            "The energy of decomposition of this material into the ste of most stable materials at this chemical
             composition, in eV/atom. Stability is tested against all potential chemical combinations that result in the
             material's composition. For example, a Co2O3 structure would be tested for decomposition against other
             Co2O3 structures, against Co and O2 mixtures, and against CoO and O2 mixtures.
             A positive E above hull indicates that this material is unstable with respect to decomposition. A zero
             E above hull indicates that this is the most stable material at its composition."

    Args:
        material_id (str): A string representing the Materials Project material ID.
        pymatgen_rester (pymatgen.ext.matproj.MPRester): An object containing the PyMatGen Materials Project RESTful
                                                         API interface implementation.

    Returns:
        The energy above hull in units of eV/atom. 0 Values indicate stability. Positive values indicate instability
        with respect to decomposition.
    '''
    entry = pymatgen_rester.get_entries(material_id)
    energy = pymatgen_rester.get_stability(entry)[0]['e_above_hull']
    return energy

if __name__ == '__main__':
    matproj_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    source_id = 'mp-1095420'
    rester = pymatgen.ext.matproj.MPRester(
        api_key=matproj_key
    )
