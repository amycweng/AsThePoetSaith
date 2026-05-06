from bs4 import BeautifulSoup, SoupStrainer,NavigableString
import re,os,subprocess
from tqdm import tqdm 
import pandas as pd 
from APS_adorn import adorn 
# output_folder = f"../"
output_folder = f"/Users/amycweng/My Drive (aw3029@princeton.edu)/DH/POETRY"

TCP = f'/Users/amycweng/DH/TCP'

def findTextTCP(id):
    if re.match('B1|B4',id[0:2]):
        path = f'{TCP}/P2{id[0:2]}/{id}.P4.xml'
    else: 
        if f'{id}.P4.xml' in os.listdir(f'{TCP}/P1{id[0:2]}'):
            path = f'{TCP}/P1{id[0:2]}/{id}.P4.xml'
        elif f'{id}.P4.xml' in os.listdir(f'{TCP}/P2{id[0:2]}'): 
            path = f'{TCP}/P2{id[0:2]}/{id}.P4.xml'
    return path 

def format_name(item):
    item = "\^".join(item.split(" "))
    item = re.sub(r"[^\d\^\\\'\w1]","\^",item)
    return item 

def extract(tcpID, filepath):
    # read the input XML file 
    with open(filepath,'r') as file: 
        data = file.read()
    # use soupstrainer to only parse the main body
    tag = SoupStrainer("DIV1")
    soup = BeautifulSoup(data,features="xml",parse_only=tag)
    contents = soup.find_all(['DIV1', 'DIV2', 'DIV3', 'DIV4', 'DIV5', 'DIV6', 'DIV7'])
    valid_divs = []

    div_text = []
    last_added_section = None 
    div_path = []
    is_Extracted = []
    for div in contents:
        
        section_name = div.name 
        section_note = "" 
        if "N" in div.attrs: 
            section_note = div["N"]
            section_note = format_name(section_note)
        section_type = div.get("TYPE").lower()
        section_type_output = format_name(section_type) 
        section_type = "_".join(section_type.split(" "))

        toAdd = True
        # if re.search("table|index|errata|list",section_type): 
        #     toAdd = False
        if section_name == "DIV1": 
            div_path = []
            is_Extracted = []
            last_added_section = None 

        for item,is_e in zip(div_path,is_Extracted): 
            d, name = item 
            # if re.search("table|index|errata|list",name): 
            #     toAdd = False
            if is_e and int(d[-1]) < int(section_name[-1]): 
                toAdd = False  
        
        div_path.append((section_name, section_type))
        is_Extracted.append(toAdd)

        if toAdd: 
            last_added_section = section_name
            text = []
            # text = re.sub(r"[^\x00-\x7F]","",str(div.text))
            for gap in div.find_all("GAP"):
                if gap["DESC"] == "foreign": 
                    gap.string = " NONLATINALPHABET " # 〈 in non-Latin alphabet 〉
                elif gap["DESC"] == "missing":
                    if "EXTENT" in gap: 
                        missing_gap = "^".join(gap["EXTENT"].upper().split(" ")) + "^MISSING"
                    else: 
                        missing_gap = "PAGES^MISSING"
                    gap.string = f' {missing_gap} '
                elif gap["DESC"] == "illegible": 
                    if "DISP" in gap.attrs: 
                        disp = gap["DISP"]
                        if "page" in disp: 
                            disp = re.sub(" ","^",disp)
                            disp = f' {disp.upper()+"^ILLEGIBLE"} '
                        # disp = re.sub("•","\^",disp) # illegible letters 
                        disp = re.sub("◊","\*",disp) # illegible words 
                        gap.string = gap["DISP"]
            for italics in div.find_all("HI"):
                italics.string = f" STARTITALICS {italics.text} ENDITALICS "
            for item in div.find_all(["NOTE"]):
                if item.name == "NOTE":
                    # add note delimiters 
                    item.string = f" STARTNOTE {item.text} ENDNOTE "
                    # item.string = " "
            for t in div.find_all(['L']):
                t.string = f" STARTLINE {t.text} ENDLINE "
            for t in div.find_all(['HEAD']):
                t.string = f" STARTHEAD {t.text} ENDHEAD "
            for t in div.find_all(['P']):
                t.string = f" STARTPARAGRAPH {t.text} ENDPARAGRAPH "
                
            # for descendant in list(div.descendants):
            #     # If it's a text node (NavigableString)
            #     if isinstance(descendant, NavigableString):
            #         parent = descendant.parent
            #         if parent.name not in ['L', 'HEAD']:
            #             descendant.replace_with('')  # Remove text not in L or HEAD
            for child in div.children:
                if child.name in ['DIV1', 'DIV2', 'DIV3', 'DIV4', 'DIV5', 'DIV6', 'DIV7']:
                    ss_type = child.get("TYPE").lower()
                    ss_type = format_name(ss_type)
                    ss_N = ""
                    if "N" in child.attrs: 
                        ss_N = child["N"]
                        ss_N = format_name(ss_N)
                    if re.search("table|errata|index|list",ss_type): 
                        text.append(f" {child.name}^{ss_type}^{ss_N} ")
                        continue  
                    text.append(f" {child.name}^{ss_type}^{ss_N} " + child.get_text() + " ")
                else: 
                    text.append(" " + child.get_text() + " ")
        
            text = f" {f' {section_name}^{section_type_output}^{section_note}'} {' '.join(text).strip()} "
            text = re.sub(r"|","",text)
            div_text.append(text)
            valid_divs.append(f"{section_name}: {section_type_output}")
        else: 
            if last_added_section is None or (int(last_added_section[-1]) > int(section_name[-1])): 
                # print(f' {section_name}^{section_type_output}^{section_note} ')
                div_text.append(f' {section_name}^{section_type_output}^{section_note} ')
    
    with open(f"{output_folder}/APS_PLAIN/{tcpID}.txt","w+") as file:
        div_text = re.sub(r"\∣|\¦|\|",""," ".join(div_text).strip())
        div_text = re.sub(r"\‖|¶|〈|〉"," ",div_text)
        div_text = re.sub(r"\s+"," ",div_text)
        file.writelines(div_text) # write as one long string 
    return "; ".join(valid_divs), len(valid_divs)   


if __name__ == "__main__": 
    # tcpIDs = list(pd.read_csv(f"APS_METADATA_ground.csv")['id'])
    tcpIDs = list(pd.read_csv(f"APS_METADATA_unknown.csv")['id'])
    half = len(tcpIDs) // 2 
    tcpIDs = sorted(tcpIDs)
    tcpIDs = tcpIDs[:half] 
    # tcpIDs = tcpIDs[half:]
    folder = f"/Users/amycweng/DH"
    already_adorned = os.listdir(f"{folder}/APS_PLAIN")
    already_adorned = {x.split(".txt")[0] for x in already_adorned}
    progress = tqdm(tcpIDs)
    for tcpID in progress: 
        progress.set_description(tcpID)
        if tcpID in already_adorned: continue 
        fp = findTextTCP(tcpID)
        extract(tcpID, fp)
        adorn(tcpID)


