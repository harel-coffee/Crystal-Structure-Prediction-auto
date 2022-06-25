per_page = 10000 # The maximum allowed data size requested using NOMAD's api

elements = pd.read_csv("Electronegativity.csv")['Symbol'][:118].to_list() # Read the list of elements
crystal_systems = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal'] 

loop = 0
total = 0

skip = 37

# The three loops serve to divide the searched data into chunks as to not exceed the 10,000 search limit, resulting in lost data

for a in range(len(elements)-1-skip):
  extracted_data = {'atoms': [], 'chemical_formula_descriptive': [], 'chemical_formula_anonymous': [], 'formula_reduced': [], 
                  'bravais_lattice': [], 'crystal_system': [], 'space_group_international_short_symbol': [], 'space_group_number': [], 
                  'point_group': [], 'lattice_parameters': [], 'basis_set': [], 'xc_functional': [], 'external_db': []}
  for b in range(len(elements)-1-a-skip):
    element_a = elements[a+skip]
    element_b = elements[b+a+skip+1]
    clear_output()
    print("Elements:", element_a, element_b)

    for cs in crystal_systems:
      response = requests.post('http://nomad-lab.eu/prod/rae/api/repo/', json={
          'query': {'dft.compound_type': 'ternary', 'atoms': [element_a, element_b], 'dft.crystal_system': cs},
          'pagination': {'page': 1, 'per_page': per_page}})

      data = response.json()

      data_size = len(data['results'])
      clear_output()
      print("Extracted Data:", total, '/ 5,966,318')
      print("Loop:", loop, '/ 48321')
      loop+=1
      total+=data_size

      if data_size == 10000:
        print("WARNING!")

      for i in range(data_size): 
        
        try:
          results = data['results'][i]
        except:
          pass

        try:
          encyclopedia = results['encyclopedia']['material']
        except:
          pass

        try:
          atoms = results['atoms']
        except:
          atoms = 'unavailable'

        try:
          chemical_formula_descriptive = results['dft']['optimade']['chemical_formula_descriptive']
        except:
          chemical_formula_descriptive = 'unavailable'

        try:
          chemical_formula_anonymous = results['dft']['optimade']['chemical_formula_anonymous']
        except:
          chemical_formula_anonymous = 'unavailable'

        try:
          formula_reduced = encyclopedia['formula_reduced']
        except:
          formula_reduced = 'unavailable'

        try:
          bravais_lattice = encyclopedia['bulk']['bravais_lattice']
        except:
          bravais_lattice = 'unavailable'

        try:
          lattice_parameters = encyclopedia['idealized_structure']['lattice_parameters']
        except:
          lattice_parameters = 'unavailable'

        try:
          crystal_system = encyclopedia['bulk']['crystal_system']
        except:
          crystal_system = 'unavailable'

        try:
          space_group_international_short_symbol = encyclopedia['bulk']['space_group_international_short_symbol']
        except:
          space_group_international_short_symbol = 'unavailable'

        try:
          space_group_number = encyclopedia['bulk']['space_group_number']
        except:
          space_group_number = 'unavailable'

        try:
          point_group = encyclopedia['bulk']['point_group']
        except:
          point_group = 'unavailable'

        try:
          basis_set = results['dft']['basis_set']
        except:
          basis_set = 'unavailable'

        try:
          xc_functional = results['dft']['xc_functional']
        except:
          xc_functional = 'unavailable'

        try:
          external_db = results['external_db']
        except:
          external_db = 'unavailable'

        for key in extracted_data:
          extracted_data[key].append(locals()[key])

  pd.DataFrame(data=extracted_data).to_pickle("/content/drive/MyDrive/CMG - Crystal Prediction Project/Ternary Materials Point Group Prediction/Data/NOMAD_2/"+str(a+1+skip)+".pkl")
