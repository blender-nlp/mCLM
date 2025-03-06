# General info

Modulizer cuts drugs (or any given compounds) into blocks compatible with Burke's lego-like automated synthesis.
The program requires the RDKit library.

# Usage

Running modulizer.py with default options cuts kinases into blocks compatible with LEGO-approach and includes bioisosteric replacements of some groups incompatible with conditions of automatic synthesis. In order to turn off bioisosteric replacements of aryl halogens to methyls, use --replaceMode noReplace , in order to turn off other bioisosteric replacements, use --initprotdb init_protection_without_bioisosteric.csv  and --finalprotdb final_protection_without_bioisosteric.csv .

In order to modularize more complex drugs, use mode alldrugs (--mode alldrugs)
In order to include Suzuki coupling aryl-alkyl and vinyl-alkyl, use mode with_suzuki_sp3_sp2 (--mode with_suzuki_sp3_sp2)

Molecules for modularization can be introduced from file via --input option (give name of the file) or from SMILES, using --smiles option (provide 'list.of.smiles.separated.by.dots')

Results will be saved automatically under "results_from_modulizer.csv", in order to change the name of the output file, use the --output option.

Users can provide their own database of reactions using --retrodb option but mind that all remaining db files have to match the reaction database, especially incolist, protdb and deprotdb
