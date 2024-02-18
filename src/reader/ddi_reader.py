import os
import re
import xmltodict

class DDICorpusXMLReader:
    def __init__(self, path):
        self.path = path
        self.train_path = os.path.join(self.path, 'Train')
        self.test_path = os.path.join(self.path, 'Test')
    
    def read_folder(self, 
                    dataset: str):
        '''
        Read XML files of DDI Corpus, return a list of dictionaries,
        each is a document.

        Args:
            dataset (str): 'train' | 'ddiextraction_test' | 'drugner_test'.
        
        Returns:
            drugbank_list (list): a list of dictionaries from drugbank, each is a document.
            medline_list (list): a list of dictionaries from medline, each is a document.
        '''
        if dataset == 'train':
            drugbank_path = os.path.join(self.train_path, 'DrugBank')
            medline_path = os.path.join(self.train_path, 'MedLine')
        elif dataset == 'ddiextraction_test':
            drugbank_path = os.path.join(self.test_path, 'DDIExtraction/DrugBank')
            medline_path = os.path.join(self.test_path, 'DDIExtraction/MedLine')
        elif dataset == 'drugner_test':
            drugbank_path = os.path.join(self.test_path, 'DrugNER/DrugBank')
            medline_path = os.path.join(self.test_path, 'DrugNER/MedLine')
        
        drugbank_list = list()
        medline_list = list()
        
        xml_detect = re.compile('\\.xml$')
            
        for file_path in sorted(os.listdir(drugbank_path)):
            if xml_detect.search(file_path):
                file_abs_path = os.path.join(drugbank_path, file_path)
                with open(file_abs_path, 'r') as file:
                    xml_file = file.read()
                    file_dict = xmltodict.parse(xml_file)
                drugbank_list.append(file_dict)
        
        for file_path in sorted(os.listdir(medline_path)):
            if xml_detect.search(file_path):
                file_abs_path = os.path.join(medline_path, file_path)
                with open(file_abs_path, 'r') as file:
                    xml_file = file.read()
                    file_dict = xmltodict.parse(xml_file)
                medline_list.append(file_dict)
        
        return drugbank_list, medline_list