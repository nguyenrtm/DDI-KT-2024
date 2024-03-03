import os
import re
import xmltodict

class TACXMLReader:
    def __init__(self, path):
        self.path = path
    
    def read_folder(self):
        '''
        Read XML files of TAC, return a list of dictionaries,
        each is a document.
        
        Returns:
            full_list: a list of dictionaries from medline, each is a document.
        '''
        full_list = list()
        
        xml_detect = re.compile('\\.xml$')

        for file_path in sorted(os.listdir(self.path)):
            if xml_detect.search(file_path):
                file_abs_path = os.path.join(self.path, file_path)
                with open(file_abs_path, 'r') as file:
                    print(file_abs_path)
                    xml_file = file.read()
                    file_dict = xmltodict.parse(xml_file)
                full_list.append(file_dict)
        
        return full_list