


import re

def extract_mol_content(text):
    pattern = re.compile(r'\[MOL\](.*?)\[/MOL\]', re.DOTALL)
    mol_list = pattern.findall(text)  # Extract MOL contents
    mol_list = [m.strip() for m in mol_list]

    cleaned_text = pattern.sub('[MOL][/MOL]', text)  # Remove MOL content from text
    return mol_list, cleaned_text.strip()

# Example input
if __name__ == '__main__':
    input_text = """[MOL] [1*]Oc1ccnc2cc(OC)c(OC)cc12.[2*]c1ccc([1*])cc1.[2*]NC(=O)C1(C(=O)N[1*])CC1.[2*]c1ccc(F)cc1 [/MOL] was associated with a significant improvement in ORR, as assessed by investigator review. Complete or partial responses were confirmed in 36 patients (46%; 95% CI, 34% to 57%) in the [MOL] [1*]Oc1ccnc2cc(OC)c(OC)cc12.[2*]c1ccc([1*])cc1.[2*]NC(=O)C1(C(=O)N[1*])CC1.[2*]c1ccc(F)cc1 [/MOL] group compared with 14 patients (18%; 95% CI, 10% to 28%) in the [MOL] [1*]C(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C.[2*]NCCN(CC)CC [/MOL] group (Table 2) . A best response of stable disease occurred in 26 patients (33%) with ascopubs.org/journal/jco [MOL] [1*]Oc1ccnc2cc(OC)c(OC)cc12.[2*]c1ccc([1*])cc1.[2*]NC(=O)C1(C(=O)N[1*])CC1.[2*]c1ccc(F)cc1 [/MOL] versus 28 patients (36%) with [MOL] [1*]C(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C.[2*]NCCN(CC)CC [/MOL], and progressive disease as best response occurred in 14 patients (18%) with [MOL] [1*]Oc1ccnc2cc(OC)c(OC)cc12.[2*]c1ccc([1*])cc1.[2*]NC(=O)C1(C(=O)N[1*])CC1.[2*]c1ccc(F)cc1 [/MOL] versus 20 patients (26%) with [MOL] [1*]C(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C.[2*]NCCN(CC)CC [/MOL]."""

    mol_list, cleaned_text = extract_mol_content(input_text)

    print("Extracted MOL contents:")
    print(mol_list)
    print("\nCleaned text:")
    print(cleaned_text)







