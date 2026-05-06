import wptools
import json 
import pandas as pd 

fields = ['description','what','aliases']
wikidata_labels = {'P244': 'Library of Congress authority ID',
                   'P214': 'VIAF cluster ID',
                    'P1415': 'Oxford Dictionary of National Biography ID',
                    'P1417': 'Encyclopædia Britannica Online ID',
        'P6886': 'writing language',
        'P569': 'date of birth',
        'P570': 'date of death',
        'P1196': 'manner of death',
        'P509': 'cause of death',
        'P27': 'country of citizenship',
        'P106': 'occupation',
        'P1066': 'student of',
        'P39': 'position held',
        'P101': 'field of work',
        'P140': 'religion or worldview',
        'P21': 'sex or gender',
        'P69': 'educated at',
        'P708': 'diocese',
        'P1343': 'described by source'
        }

def get_dnb_link(page): 
    parsed = page.get_parse()
    external_links = parsed.data["iwlinks"]
    if external_links: 
        for link in external_links: 
            if "Dictionary_of_National_Biography" in link: 
                return link 
    return None 

def get_wiki(linked_entry): 
    wiki_id = linked_entry[3]
    info_dict = {
        'TCP_name_heading':linked_entry[0],
        'entity_type':linked_entry[1],
        'entity_title':linked_entry[2],
        'wiki_entity_id':wiki_id
    }
    for field in fields: 
        info_dict[field] = ''
    for field_id, field_name in wikidata_labels.items():   
        info_dict[field_name] = ''

    if not wiki_id: 
        return info_dict
    try:
        page = wptools.page(wikibase=wiki_id)
        info = page.get_wikidata()
    except Exception:
        return info_dict

    for field in fields: 
        if field in info.data: 
            info_dict[field] = info.data[field]
            # print(field,info.data[field])

    for field_id, field_name in wikidata_labels.items():   
        wikidata = info.data['wikidata'] 
        key = f"{field_name} ({field_id})"
        info_dict[field_name] = ''
        if key in wikidata: 
            info_dict[field_name] = wikidata[key]
            # print(field_name, wikidata[key])
    info_dict["Oxford DNB Link"] = get_dnb_link(page)
    return info_dict

if __name__=="__main__": 
    with open("../features/APS_authors_linked.json","r") as file: 
        linked_authors = json.load(file)
    info = []
    fname = "../features/APS_authors_wiki.csv"

    for idx, entry in enumerate(linked_authors): 
        print(idx,"out of",len(linked_authors))
        if idx % 250 == 0: 
            info_df = pd.DataFrame(info)
            print(info_df)
            info_df.to_csv(fname,index=False)
        if isinstance(entry[-1],float): continue 
        entry = get_wiki(entry)
        # if entry['what'] == "human": 
        info.append(entry)
        
    info_df = pd.DataFrame(info)
    print(info_df)
    info_df.to_csv(fname,index=False)

        
