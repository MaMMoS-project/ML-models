import re
import numpy as np

# CONVERSION UNITS
MU_NOUGHT = 4e-7 * np.pi # N/A^2
MU_B = 9.2740100657e-24 # J⋅T^−1
AA = 1e-10 # m

# List of periodic table elements
pd_elements_list = [
    'H',  # Hydrogen
    'He',  # Helium
    'Li',  # Lithium
    'Be',  # Beryllium
    'B',  # Boron
    'C',  # Carbon
    'N',  # Nitrogen
    'O',  # Oxygen
    'F',  # Fluorine
    'Ne',  # Neon
    'Na',  # Sodium
    'Mg',  # Magnesium
    'Al',  # Aluminum
    'Si',  # Silicon
    'P',  # Phosphorus
    'S',  # Sulfur
    'Cl',  # Chlorine
    'Ar',  # Argon
    'K',  # Potassium
    'Ca',  # Calcium
    'Sc',  # Scandium
    'Ti',  # Titanium
    'V',  # Vanadium
    'Cr',  # Chromium
    'Mn',  # Manganese
    'Fe',  # Iron
    'Co',  # Cobalt
    'Ni',  # Nickel
    'Cu',  # Copper
    'Zn',  # Zinc
    'Ga',  # Gallium
    'Ge',  # Germanium
    'As',  # Arsenic
    'Se',  # Selenium
    'Br',  # Bromine
    'Kr',  # Krypton
    'Rb',  # Rubidium
    'Sr',  # Strontium
    'Y',  # Yttrium
    'Zr',  # Zirconium
    'Nb',  # Niobium
    'Mo',  # Molybdenum
    'Tc',  # Technetium
    'Ru',  # Ruthenium
    'Rh',  # Rhodium
    'Pd',  # Palladium
    'Ag',  # Silver
    'Cd',  # Cadmium
    'In',  # Indium
    'Sn',  # Tin
    'Sb',  # Antimony
    'Te',  # Tellurium
    'I',  # Iodine
    'Xe',  # Xenon
    'Cs',  # Cesium
    'Ba',  # Barium
    'La',  # Lanthanum
    'Ce',  # Cerium
    'Pr',  # Praseodymium
    'Nd',  # Neodymium
    'Pm',  # Promethium
    'Sm',  # Samarium
    'Eu',  # Europium
    'Gd',  # Gadolinium
    'Tb',  # Terbium
    'Dy',  # Dysprosium
    'Ho',  # Holmium
    'Er',  # Erbium
    'Tm',  # Thulium
    'Yb',  # Ytterbium
    'Lu',  # Lutetium
    'Hf',  # Hafnium
    'Ta',  # Tantalum
    'W',  # Tungsten
    'Re',  # Rhenium
    'Os',  # Osmium
    'Ir',  # Iridium
    'Pt',  # Platinum
    'Au',  # Gold
    'Hg',  # Mercury
    'Tl',  # Thallium
    'Pb',  # Lead
    'Bi',  # Bismuth
    'Po',  # Polonium
    'At',  # Astatine
    'Rn',  # Radon
    'Fr',  # Francium
    'Ra',  # Radium
    'Ac',  # Actinium
    'Th',  # Thorium
    'Pa',  # Protactinium
    'U',  # Uranium
    'Np',  # Neptunium
    'Pu',  # Plutonium
    'Am',  # Americium
    'Cm',  # Curium
    'Bk',  # Berkelium
    'Cf',  # Californium
    'Es',  # Einsteinium
    'Fm',  # Fermium
    'Md',  # Mendelevium
    'No',  # Nobelium
    'Lr',  # Lawrencium
    'Rf',  # Rutherfordium
    'Db',  # Dubnium
    'Sg',  # Seaborgium
    'Bh',  # Bohrium
    'Hs',  # Hassium
    'Mt',  # Meitnerium
    'Ds',  # Darmstadtium
    'Rg',  # Roentgenium
    'Cn',  # Copernicium
    'Nh',  # Nihonium
    'Fl',  # Flerovium
    'Mc',  # Moscovium
    'Lv',  # Livermorium
    'Ts',  # Tennessine
    'Og'  # Oganesson
]

pd_elements_set = set(pd_elements_list)


# RARE EARTHS
RARE_EARTHS_LIST =  ['Sc', 'Y', 'La', 'Ce', 'Pr',  \
                   'Nd', 'Pm', 'Sm', 'Eu', 'Gd', \
                   'Tb', 'Dy', 'Ho', 'Er', 'Tm', \
                   'Yb', 'Lu']
RARE_EARTHS = set( RARE_EARTHS_LIST )

TOKENIZED_RARE_EARTHS = { k:v for k,v in enumerate(sorted(RARE_EARTHS_LIST)) }


# COMPOUND TO ELEMENT CONTENT
def compound_2_element_content(compound):
    """
    Signature:
    compound_2_element_content(compound)
    
    Docstring:
    Method that returns the dictionary { element : content  } for a given compound formula. Compounds containing polyatomic ions are also allowed.
    
    Parameters:
    compound : str
    
    Examples:
    In[0]:  compound_2_element_content(Nd2Fe14B)
    Out[0]: { 'Nd': 2, 'Fe': 14, 'B': 1  }
    
    In[1]:  compound_2_element_content(Pr2Fe11.2Mn0.8Co2B)
    Out[1]: { 'Pr': 2, 'Fe': 11.2, 'Mn': 0.8, 'Co' : 2, 'B' : 1  }
    
    In[2]:  compound_2_element_content('X3Y2(Ab2Cd2E)2Z.6')
    Out[2]: {'X': 3.0, 'Y': 2.0, 'Ab': 4.0, 'Cd': 4.0, 'E': 2.0, 'Z': 0.6}
    
    Returns:
    dict
    
    """
    def compound_2_element_content_no_brackets(compound):
        
        # check if element belongs to the periodic table. If it does, update element_content dict.
        def update_element_content(el_cont, el, cont):
            if el in pd_elements_set:
                el_cont[el] = float(cont)
            else:
                raise ValueError('{} does not belong to the periodic table!'.format(el))
            
            
        element_content = {}
        element = ''
        content = ''
    
        previous_char = ''
    
    
        for l,i in zip(compound,range(len(compound))):
            if( l.isalpha() ):
                if( not previous_char.isalpha() ):
                    if( i == 0 ):
                        element+=l
                    else:
                        update_element_content(element_content, element, content)
                        #element_content[element] = float(content)
                        element = l
                        content = ''
                        
                elif ( previous_char.isalpha() ) :
                    if (l.islower()):
                        element+=l
                    if ( l.isupper() ):
                        update_element_content(element_content, element, 1)
                        # element_content[element] = 1.
                        element = l
                        content = ''
                if(i == len(compound) -1 ):
                    update_element_content(element_content, element, 1)
                    # element_content[element] = 1.
                    
            if ( l.isnumeric() or l == '.'):
                content+=l
                # if (previous_char.isnumeric()):
                #     content+=l
                if(i == len(compound) -1 ):
                    update_element_content(element_content, element, content)
                    #element_content[element] = float(content)
                        
            
            previous_char = l
                    
        return element_content
    
    element_content = {}
    subcompounds = re.split(r'[()]', compound)

    
    if len(subcompounds) == 0:
        raise ValueError('Entered empty compound string!')\
        
    if len(subcompounds) == 1:
        return compound_2_element_content_no_brackets(subcompounds[0])

    mult_factor = ''
    
    
    for i in range(1,len(subcompounds)): # ultimo elemento puo essere un numero
        new_elements={}
        
        if (subcompounds[i-1] == ''):
            continue
        if (subcompounds[i][0].isalpha()  ):
            new_elements = compound_2_element_content_no_brackets(subcompounds[i-1])
            
        if (subcompounds[i][0].isnumeric() or subcompounds[i][0] == '.'  ):
            n = subcompounds[i][0]
            cnt = 1
            for l in subcompounds[i][1:]:
                if(l.isnumeric() or l =='.'):
                    n+=l
                    cnt+=1
                else:
                    break
            subcompounds[i] = subcompounds[i][cnt:]
            mult_factor+=n

    
            new_elements = compound_2_element_content_no_brackets(subcompounds[i-1])
            for k,v in new_elements.items():
                new_elements[k] *= float(mult_factor)
            mult_factor = ''
        
        
        if (i==len(subcompounds)-1 and subcompounds[-1]!='' and \
            (subcompounds[-1][0]=='.' or subcompounds[-1][0].isnumeric() ) ):

            
            new_elements = compound_2_element_content_no_brackets(subcompounds[i-1])
            
            for k,v in new_elements.items():
                new_elements[k] *= float(mult_factor)
            mult_factor = ''
            

        element_content.update(new_elements )

    if (subcompounds[-1] !='' and len(subcompounds) > 1 and subcompounds[-1][0].isalpha()):
        element_content.update(compound_2_element_content_no_brackets(subcompounds[-1]) )
        
        
                
    return element_content



def rare_earth_content(compound):
    """
    Signature:
    rare_earth_content(compound)
    
    Docstring:
    Method that returns a set whose element are the rare earth elements in the compound. An empty set is returned if the compound in in put is rare-earth-free
    
    Parameters:
    compound : str
    
    Examples:
    
    Returns:
    set
    
    """
    
    return set(compound_2_element_content(compound).keys()).intersection(RARE_EARTHS)

def has_rare_earth(compound):
    """
    Signature:
    rare_earth_content(compound)
    
    Docstring:
    Method that returns True (False) if the compound in input contains rare earths (is rare-earth-free).
    
    Parameters:
    compound : str
    
    Examples:
    
    Returns:
    bool_
    
    """
    
    return bool(rare_earth_content(compound))