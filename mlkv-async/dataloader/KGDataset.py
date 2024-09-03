# -*- coding: utf-8 -*-
#
# KGDataset.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import numpy as np

def _download_and_extract(url, path, filename):
    import shutil, zipfile
    import requests

    fn = os.path.join(path, filename)

    while True:
        try:
            with zipfile.ZipFile(fn) as zf:
                zf.extractall(path)
            print('Unzip finished.')
            break
        except Exception:
            os.makedirs(path, exist_ok=True)
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(fn, 'wb') as writer:
                for chunk in f_remote.iter_content(chunk_size=1024*1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')

class KGDataset:
    '''Load a knowledge graph

    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * test stores the triples in the test set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''
    def __init__(self, entity_path, relation_path, train_path,
                 valid_path=None, test_path=None, format=[0,1,2],
                 delimiter='\t', skip_first_line=False):
        self.delimiter = delimiter
        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line, format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", skip_first_line, format)
        else:
            self.test = None

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split(self.delimiter)
                entity2id[entity] = int(eid)

        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split(self.delimiter)
                relation2id[relation] = int(rid)

        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=[0,1,2]):
        # mode: train/valid/test
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return (heads, rels, tails)

class KGDatasetFB15k(KGDataset):
    '''Load a knowledge graph FB15k

    The FB15k dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='FB15k'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetFB15k, self).__init__(os.path.join(self.path, 'entities.dict'),
                                             os.path.join(self.path, 'relations.dict'),
                                             os.path.join(self.path, 'train.txt'),
                                             os.path.join(self.path, 'valid.txt'),
                                             os.path.join(self.path, 'test.txt'))

    @property
    def emap_fname(self):
        return 'entities.dict'

    @property
    def rmap_fname(self):
        return 'relations.dict'

class KGDatasetFreebase(KGDataset):
    '''Load a knowledge graph Full Freebase
    The Freebase dataset has five files:
    * entity2id.txt stores the mapping between entity name and entity Id.
    * relation2id.txt stores the mapping between relation name relation Id.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.
    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='Freebase'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, '{}.zip'.format(name))
        self.path = os.path.join(path, name)

        super(KGDatasetFreebase, self).__init__(os.path.join(self.path, 'entity2id.txt'),
                                                os.path.join(self.path, 'relation2id.txt'),
                                                os.path.join(self.path, 'train.txt'),
                                                os.path.join(self.path, 'valid.txt'),
                                                os.path.join(self.path, 'test.txt'))

    def read_entity(self, entity_path):
        with open(entity_path) as f_ent:
            n_entities = int(f_ent.readline()[:-1])
        return None, n_entities

    def read_relation(self, relation_path):
        with open(relation_path) as f_rel:
            n_relations = int(f_rel.readline()[:-1])
        return None, n_relations

    def read_triple(self, path, mode, skip_first_line=False, format=None):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                h, t, r = line.strip().split(self.delimiter)
                heads.append(int(h))
                tails.append(int(t))
                rels.append(int(r))

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        return (heads, rels, tails)

    @property
    def emap_fname(self):
        return 'entity2id.txt'

    @property
    def rmap_fname(self):
        return 'relation2id.txt'

class KGDatasetWikikg2(KGDataset):
    '''Load a knowledge graph wikikg2

    The wikikg2 dataset has five files:
    * entities.dict stores the mapping between entity Id and entity name.
    * relations.dict stores the mapping between relation Id and relation name.
    * train.txt stores the triples in the training set.
    * valid.txt stores the triples in the validation set.
    * test.txt stores the triples in the test set.

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'.
    '''
    def __init__(self, path, name='wikikg2'):
        self.name = name
        url = 'https://data.dgl.ai/dataset/{}.zip'.format(name)

        if not os.path.exists(os.path.join(path, name)):
            print('File not found. Downloading from', url)
            _download_and_extract(url, path, name + '.zip')
        self.path = os.path.join(path, name)

        super(KGDatasetWikikg2, self).__init__(os.path.join(self.path, 'entities.dict'),
                                              os.path.join(self.path, 'relations.dict'),
                                              os.path.join(self.path, 'train.txt'),
                                              os.path.join(self.path, 'valid.txt'),
                                              os.path.join(self.path, 'test.txt'))

    @property
    def emap_fname(self):
        return 'entities.dict'

    @property
    def rmap_fname(self):
        return 'relations.dict'

def get_dataset(data_path, data_name, format_str, delimiter='\t', files=None, has_edge_importance=False):
    if format_str == 'built_in':
        if data_name == 'Freebase':
            dataset = KGDatasetFreebase(data_path)
        elif data_name == 'FB15k':
            dataset = KGDatasetFB15k(data_path)
        elif data_name == 'FB15k-237':
            dataset = KGDatasetFB15k237(data_path)
        elif data_name == 'wn18':
            dataset = KGDatasetWN18(data_path)
        elif data_name == 'wn18rr':
            dataset = KGDatasetWN18rr(data_path)
        elif data_name == 'wikikg2':
            dataset = KGDatasetWikikg2(data_path)
        elif data_name == 'biokg':
            dataset = KGDatasetBiokg(data_path)
        elif data_name == 'wikikg90M':
            dataset = KGDatasetWiki90M(data_path)
        else:
            assert False, "Unknown dataset {}".format(data_name)
    elif format_str.startswith('raw_udd'):
        # user defined dataset
        assert data_name != 'FB15k', 'You should provide the dataset name for raw_udd format.'
        format = format_str[8:]
        dataset = KGDatasetUDDRaw(data_path, data_name, delimiter, files, format, has_edge_importance)
    elif format_str.startswith('udd'):
        # user defined dataset
        assert data_name != 'FB15k', 'You should provide the dataset name for udd format.'
        format = format_str[4:]
        dataset = KGDatasetUDD(data_path, data_name, delimiter, files, format, has_edge_importance)
    else:
        assert False, "Unknown format {}".format(format_str)

    return dataset
