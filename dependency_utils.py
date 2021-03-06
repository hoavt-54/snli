# -*- coding: utf-8 -*-
import json
import os.path
from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = '/users/ud2017/hoavt/stanford_corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
path_to_models_jar = '/users/ud2017/hoavt/stanford_corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
#dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
from collections import defaultdict
class Parser(object):
    def __init__(self, datasetName, path_to_models_jar=path_to_models_jar, path_to_jar=path_to_jar, path_to_save='/users/ud2017/hoavt/nli/BiMPM/models'):
        self.dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar, java_options='-mx20000m')
        self.path_to_save = path_to_save
        self.cache = {}
        self.datasetName = datasetName
        self.load_cache()
        #types = "acomp advcl advmod agent amod appos aux auxpass cc ccomp conj cop csubj csubjpass \
        #        dep det discourse dobj expl goeswith iobj mark mwe neg nn npadvmod nsubj nsubjpass \
        #        num number parataxis pcomp pobj poss possessive preconj predet prep prepc prt punct \
        #        quantmod rcmod ref root tmod vmod xcomp xsubj nmod"
        types = "acl acl:relcl advcl advmod amod appos aux auxpass case cc cc:preconj ccomp compoun \
                compound:prt conj cop csubj csubjpass dep det det:predet discourse dislocated dobj \
                expl foreign goeswith iobj list mark mwe name neg nmod nmod:npmod nmod:poss nmod:tmod \
                nsubj nsubjpass nummod parataxis punct remnant reparandum root vocative xcomp compound";
        
        self.type2idx = defaultdict(lambda: len(self.type2idx))
        for t in types.strip().split():
            self.type2idx[t.strip()]
        self.typesize = len(self.type2idx)
        print "typesize: ", self.typesize 
    def isParsed(self,sentence):
        return self.cache and sentence in self.cache
    def parse_sentences(self, sentences):
        results = self.dependency_parser.raw_parse_sents(sentences)
        results = list(results)
        for idx, result in enumerate(results):
            self.parse(sentences[idx], list(result)[0])

     
    def parse(self, sentence, result=None):
        if sentence in self.cache:
            return self.cache[sentence]
        print 'not found in cache: ', sentence
        if not result:
            result = self.dependency_parser.raw_parse(sentence)
            dep_res = result.next()
            nodes = dep_res.nodes
        else:
            nodes = result.nodes
        parsed_sent = self.emptylistmaker(len(sentence.split())) #[[0...0],[0...0], ...]
        dep_cons = self.neglistmaker(len(sentence.split())) #[-1, -1 ... -1]
        #print nodes, len(nodes), len(parsed_sent), len(sentence.split())
        for idx in range(len(nodes)):
            try:
                node = nodes[idx]
                if idx == 0:
                    dep_idx = node['deps']['root'][0]
                    dep_type_idx = self.type2idx['root']
                    root = parsed_sent[dep_idx - 1]
                    root[dep_type_idx] = 1
                    parsed_sent[dep_idx - 1] = root
                    # for connection
                    dep_cons[dep_idx - 1] = -1
                    continue
                head = parsed_sent[idx-1]
                for dep in node['deps']: # nsubj: [5]
                    try:
                        dep_type_idx = self.type2idx[dep]
                        dep_idx = node['deps'][dep][0]
                        #print 'word:', node['word'], 'idx:', idx, 'type:', dep, 'dep_type_idx:', dep_type_idx, 'dep_idx:', dep_idx
                        dependent = parsed_sent[dep_idx - 1]
                        dependent[dep_type_idx] = -1
                        head[dep_type_idx] = 1 
                        #print head
                        #print dependent
                        parsed_sent[idx-1] = head
                        parsed_sent[dep_idx - 1] = dependent
                        #add dependency connection
                        dep_cons[dep_idx - 1] = idx-1

                    except Exception as e:
                        print(list(dep_res.triples()))
                        print str(e)
                        print sentence
                        print 'word:', node['word'], 'idx:', idx, 'type:', dep, 'dep_type_idx:', dep_type_idx, 'dep_idx:', dep_idx
                        print node['deps']
                        print nodes
                        print len(nodes)
                        print len(parsed_sent)
            except  Exception as e:
                print str(e)
                print sentence
        results = {'emb':parsed_sent, 'con': dep_cons} 
        self.cache[sentence] = results
        return results
    def load_cache(self):
        print "loading dependency cache"
        #import glob
        #for jfile in glob.glob(self.path_to_save + '/' + self.datasetName + '_*.json'):
        #    print jfile
        #    with open(jfile) as f:
        #        cache = json.load(f)
        #        self.cache = dict(self.cache.items() + cache.items())
        
        if not os.path.isfile(self.path_to_save + '/' + self.datasetName +'.json'): return
        with open(self.path_to_save +'/' + self.datasetName + '.json') as f:
            self.cache = json.load(f)
            
    def save_cache(self):
        with open(self.path_to_save +'/' + self.datasetName + '.json', 'w') as outfile:
            json.dump(self.cache, outfile)

    def zerolistmaker(self, n):
        listofzeros = [0] * n
        return listofzeros
    def neglistmaker(self, n):
        listneg = [-2] *n
        return listneg
    
    def emptylistmaker(self, n):
        listofzeros = self.zerolistmaker(self.typesize)
        emptylist = []
        for x in range(n):
            emptylist.append(self.zerolistmaker(self.typesize))
        return emptylist

if __name__ == '__main__':
    import sys
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    parser = Parser('snli')
    #parser.save_cache()
    #results = parser.parse('Hoa is the most handsome guy on earth .')
    #print '\n\n'
    #print results['con']
    #for w in results: print w
    #results = parser.parse_sentences(['the blode woman is riding the bike', 'Hoa is the most handsome guy on earth'])
    #parser.save_cache()
    sys.exit()
    path = '/users/ud2017/hoavt/nli/snli_1.0/snli_1.0_train5.tsv'
    with open(path, 'r') as filein:
        count = 1
        sentences = []
        for line in filein:
            parts = line.strip().split('\t')
            print count, parts[1], parts[2]
            if not parser.isParsed(parts[1]):
                sentences.append(parts[1])
            if not parser.isParsed(parts[2]):
                sentences.append(parts[2])
            if(count > 100 and len(sentences) > 80):
                parser.parse_sentences(sentences)
                sentences = []
                parser.save_cache()
                count = 0
            count+=1
        parser.parse_sentences(sentences)
    parser.save_cache()
