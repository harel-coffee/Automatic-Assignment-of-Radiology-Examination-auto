import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from sys import platform
import argparse
import numpy as np
import pandas as pd
import pickle
from nltk.classify import MaxentClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support,classification_report
from imblearn.over_sampling import RandomOverSampler
from nltk.util import ngrams
import torch
import random
from collections import Counter
from augmentation import get_POS,augment
from model import BERT

groups = ["Group - CT CAP IV and Oral",
"Group - CT Abdomen Pelvis w IV Only",
"Group - CT CAP IV Only",
"Group - CT Abdomen Pelvis w IV and Oral",
"Group - CT Renal Mass",
"Group - CT Liver 3 Phase",
"Group - CT Abdomen Pelvis No Contrast",
"Group - CT IVP  50 yrs +",
"Group - CT CAP Oral Only",
"Group - CT CAP No Contrast ",
"Group - CT Abd Pel Enterography",
"Group - Liver 4 Phase",
"Group - CT CA IV Only",
"Group - CT IVP < 50",
"Group - CT Pancreas Mass 3 Phase",
"Group - CT Abdomen No Contrast",
"Group - CT CA IV and Oral",
"Group - CT Pelvis IV Only",
"Group - CT Abdomen IV and Oral",
"Group - CT Pancreas Mass 2 Phase",
"Group - CT Abdomen Pelvis w Oral only",
"Group - CT CA No Contrast",
"Group - CT Pelvis Cystogram",
"Group - CT Liver 2 Phase",
"Group - CT Pelvis IV and Oral"]
#    excluded
# "Group - CT CA Oral Only",
# "Group - CT Abdomen IV Only",]



class auto():

    def __init__(self, args):
        self.args=args
        self.retry=True
        self.setSeed()

    def setSeed(self):
        seed_val = self.args.seed
        print('setting seed {}'.format(seed_val))

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.benchmark = True

    def oversample(self, n=2):

        sampling_strategy={}
        for g in groups[2:]:
            sampling_strategy[g] = len(self.data.index[self.data['groupID'] == groups[1]])

        oversample = RandomOverSampler(sampling_strategy=sampling_strategy)
        self.X_train,self.Y_train = oversample.fit_resample([[x] for x in self.X_train], self.Y_train)
        self.X_train = [x[0] for x in self.X_train]

    def downsample(self, n=2):
        cnt = sorted(Counter(self.Y_train).items(),
                     key=lambda item: (-item[1]))
        train_data = pd.DataFrame(zip(self.X_train, self.Y_train))
        train_data.columns = ['X', 'Y']
        max = cnt[n][1]
        drop_idx = []
        for i in range(n):
            g = [g for g, _ in enumerate(self.Y_train) if self.Y_train[g] == groups[i]]
            if (len(g) > max):
                drop_idx.extend(random.sample(g, len(g) - max))

        train_data = train_data.drop(index=drop_idx)
        train_data.reset_index(drop=True, inplace=True)
        self.X_train  = train_data['X'].to_list()
        self.Y_train  = train_data['Y'].to_list()

    def print_size(self,data, groups):
        for g in groups:
            print(g,  len(data.index[data['groupID'] == g].tolist()))
        print('\n')


    def prep(self,fold):
        self.data = pickle.load(open('data.pkl' if self.args.type != 'nn' else 'data_nn.pkl', 'rb'))
        self.data_map ={t['text']:{ 'groupID':t['groupID']} for i,t in self.data.iterrows()}
        self.groups = groups

        if args.distill:
            self.data_distill = pickle.load(open("soft_labels_fold_{}.pickle".format(fold), 'rb'))

            if self.args.testrun:
                self.data_distill = self.data_distill[0:100]
                self.data_distill = self.data_distill+[(d['text'],self.data_distill[0][1],self.data_distill[0][2]) for i, d in self.data.iterrows()]
                self.data = self.data.groupby('groupID').apply(lambda x: x.sample(min(len(x), 25))).sample(
                    frac=1).reset_index(drop=True)
                self.data = self.data[0:100]

            self.data_distill_map = {t[0]:{'softlabel':t[1], 'groupID':t[2]} for t in self.data_distill}

            self.augmented_idx = [i for i,t in enumerate(self.data_distill) if '<mask>' in t[0]]

            doc_lens = [len(t) for t in self.data['text']]
            print(' min {:.2f} mean {:.2f} median {:.2f} max {:.2f} '.format(  np.min(doc_lens),
                                                                                       np.mean(doc_lens),
                                                                                       np.median(doc_lens),
                                                                                       np.max(doc_lens)))

        else:

            if self.args.testrun:
                self.data = self.data.groupby('groupID').apply(lambda x: x.sample(min(len(x), 25))).sample(
                    frac=1).reset_index(drop=True)
                self.data = self.data[0:100]

            data_group = self.data.copy()
            self.print_size(data_group,self.groups)

            if data_group  is not None:
                doc_lens=[len(t) for t in data_group['text']]
                print(' min {:.2f} mean {:.2f} median {:.2f} max {:.2f} '.format(np.min(doc_lens), np.mean(doc_lens ), np.median(doc_lens ), np.max(doc_lens )))

    def cv(self,fold):

        print('cv ---------- fold ', fold)
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        X = self.data['text']
        Y = self.data['groupID']
        self.Y_original  = self.data['groupID'].copy()
        self.X_train= {}
        self. X_validation = {}
        self.X_train_teacher = {}
        self.data_train = {}
        self.data_validation = {}
        self.Y_train = {}
        self.Y_train_teacher = {}
        self.Y_validation ={}
        self.Y_train_originial  = {}
        self.Y_validation_originial = {}
        self.Y_train_softlabels = {}
        self.features_coded_validation={}
        self.features_coded_training={}

        if os.path.exists('trainig_fold_{}.pickle'.format(fold)) and not self.args.testrun and self.args.type == 'nn':
            [self.X_train, self.X_validation, self.Y_train, self.Y_validation,train_index,test_index] =pickle.load(open('trainig_fold_{}.pickle'.format(fold), 'rb'))
        else:
            f = 0
            for train_index, test_index in kf.split(X, Y):
                self.X_train,  self.X_validation , self.data_train, self.data_validation = [X[i] for i in train_index], [X[i] for i in test_index], [self.data.iloc[i]['text'] for i in train_index], [self.data.iloc[i] ['text'] for i in test_index]
                self.Y_train_originial, self.Y_validation_originial = [self.Y_original[i] for i in train_index], [self.Y_original[i] for i in test_index]
                self.Y_train, self.Y_validation = [self.Y_original[i] for i in train_index], [self.Y_original[i] for i in test_index]

                f += 1
                if (f == fold+1):
                    if not args.testrun  and self.args.type == 'nn':
                        pickle.dump([self.X_train,  self.X_validation, self.Y_train, self.Y_validation,train_index,test_index], open('trainig_fold_{}.pickle'.format(fold), 'wb'))

                    break


        pkl_file ='augmented_data_nn_fold_{}.pkl'.format(fold)
        self.distribution = pd.DataFrame({'count': self.data .groupby([ 'groupID']).size()}).reset_index()
        if self.args.type =='nn' and self.args.augment and (self.args.augment_fold<0 or self.args.augment_fold==fold):
            # Generate augmented samples

            if not os.path.exists(pkl_file):
                augmented_file = 'POS_sentences_fold_{}.pkl'.format(fold)
                if not os.path.exists(augmented_file):
                    POS_map = get_POS(self.X_train)
                    pickle.dump(POS_map, open(augmented_file, 'wb'))
                else:
                    POS_map= pickle.load(open(augmented_file, 'rb'))

                self.group_augment={}
                for g in groups:
                    if self.args.augment_size <0 or (g in self.distribution['groupID'].values and self.distribution[self.distribution['groupID']==g]['count'].values[0] < self.args.augment_size):
                        group_idx = [i for i, label  in enumerate(self.Y_train) if label==g]
                        group_data = [self.X_train[i] for i in group_idx]
                        group_data = pd.DataFrame(group_data,columns=['text'] )
                        self.group_augment = augment( g, group_data, POS_map,self.args.augment_size-self.distribution[self.distribution['groupID']==g]['count'].values[0], self.args.augment_per_instance)

                pickle.dump(self.group_augment, open(pkl_file, 'wb'))

        elif self.args.type =='nn' and self.args.softlabels :

            group_augment = pickle.load(open(pkl_file, 'rb'))
            augmented_text=[]
            augmented_labels =[]
            for l in group_augment:
                for i, t in group_augment[l].iterrows():
                    augmented_text.append(t.values[0])
                    augmented_labels.append(l)

            self.df_augmented=pd.DataFrame(zip(augmented_text,augmented_labels), columns=['text','label'])
            train_df = pd.DataFrame(zip(self.X_train,self.Y_train), columns=['text','label'])
            self.df_augmented=self.df_augmented.append(train_df)
            self.df_augmented.reset_index(drop=True, inplace=True)
            print('augmented_data+data:',len(self.df_augmented))

        if args.distill:
            self.X_train_teacher,self.Y_train_teacher=self.X_train,self.Y_train
            self.X_train_teacher_num=len(self.X_train_teacher)

            self.X_train,self.Y_train_softlabels, self.Y_train =self.X_train_teacher+[self.data_distill[i][0] for i in self.augmented_idx],\
                                               [self.data_distill_map[t]['softlabel'] for t in self.X_train ]+[self.data_distill[i] for i in self.augmented_idx]\
                                                ,self.Y_train_teacher + [self.data_distill[i][2] for i in  self.augmented_idx]

            print('x_train: {} x_validation: {} X_train_teacher: {} '.format( len(self.X_train),   len(self.Y_validation)  , len(self.X_train_teacher)))

        print('x_train:',len(self.X_train),'x_validation:',len(self.Y_validation))

        if self.args.type != 'nn':
            if not self.args.nocoded :
                self.features_coded_training = self.data.iloc[train_index][['age', 'sex', 'examcode']]
                self.features_coded_validation = self.data.iloc[test_index][['age', 'sex', 'examcode']]
                self.features_coded_training.reset_index(drop=True, inplace=True)
                self.features_coded_validation.reset_index(drop=True, inplace=True)

                assert len(self.features_coded_training)==len(self.X_train)==len(self.Y_train)
                assert len(self.features_coded_validation) == len(self.X_validation) == len(self.Y_validation_originial)


            if self.args.notes:
                self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, self.args.ng),
                                           stop_words=None)


            le = preprocessing.LabelEncoder()
            le.fit(self.data["examcode"])
            if not self.args.nocoded:
                    self.features_coded_training["examcode"] = le.transform(self.features_coded_training["examcode"])
                    self.features_coded_validation["examcode"] = le.transform(self.features_coded_validation["examcode"])
                    self.features_coded_training = self.features_coded_training .apply(pd.to_numeric)
                    self.features_coded_validation = self.features_coded_validation .apply(pd.to_numeric)

            if (self.args.type in [ 'svm' ,'decisiontree','randomforest', 'gbm'] ):
                if self.args.notes:
                    self.features = self.tfidf.fit_transform(self.X_train).toarray()
                    self.features_validation = self.tfidf.transform(self.X_validation).toarray()
                    if not self.args.nocoded:
                        self.features = np.concatenate([self.features_coded_training, self.features ], axis=1)
                        self.features_validation = np.concatenate([self.features_coded_validation, self.features_validation], axis=1)
                elif not self.args.nocoded:
                    self.features = self.features_coded_training
                    self.features_validation =  self.features_coded_validation

            elif self.args.type == 'maxent':
                txt_features = [{' '.join(ng): True for ng in ngrams(tokens, self.args.ng)} for tokens in
                                [t.split() for t in self.data['text']]]
                if self.args.notes:
                    features = np.concatenate([self.features,
                                               txt_features], axis=1)

                labels = self.data['groupID']

            if args.balance :

                for g in self.groups:
                    x_pklfile = 'fold_{}_balance_{}_ngram_{}_group_{}_{}_features.pkl'.format(fold,self.args.balance,self.args.ng,g, 'coded' if not self.args.nocoded else 'uncoded')
                    y_pklfile = 'fold_{}_balance_{}_ngram_{}_group_{}_{}_y.pkl'.format(fold, self.args.balance,
                                                                                                    self.args.ng, g,
                                                                                                    'coded' if not self.args.nocoded else 'uncoded')
                    print ('pkl file ',x_pklfile, os.path.exists(x_pklfile))
                    if os.path.exists(x_pklfile):
                        self.features  = pickle.load(open(x_pklfile, 'rb'))
                        self.Y_train = pickle.load(open(y_pklfile, 'rb'))
                    else:
                        from imblearn.over_sampling import RandomOverSampler,SVMSMOTE
                        from imblearn.combine import SMOTEENN, SMOTETomek
                        if self.args.smote == 'SVMSMOTE':
                            sampler = SVMSMOTE(random_state=0)
                        else:
                            sampler = SMOTEENN(random_state=0)
                        print('re sampling ... group {} fold {} self.X_train {} self.Y_train {}'.format(g,fold,self.X_train.__len__(), self.Y_train.__len__()))
                        self.features, self.Y_train = sampler.fit_resample(self.features, self.Y_train)
                        df = pd.DataFrame(self.Y_train)
                        df.columns = ['y']
                        df['i'] = 1
                        z = pd.DataFrame(df[['i', 'y']].groupby(['y']).size())
                        print('Y_train :', z.values)

                        # if not coded:
                        pickle.dump(self.features, open(x_pklfile, 'wb'))
                        pickle.dump(self.Y_train, open(y_pklfile, 'wb'))

    def generate_softlabels(self,fold):
        self.args.nn_model = os.path.join(self.args.nn_teacher_model, 'saved_model_fold_{}'.format(fold))

        print('generating... xtrain: {} ytrain {} augmented {} '.format(len(self.X_train),len(self.Y_train),len(list(self.df_augmented['text']))))
        model = BERT(self.X_train+list(self.df_augmented['text']), self.X_validation, self.Y_train+list(self.df_augmented['label']), None, self.Y_validation, self.args,
                       self.groups)

        model.generate_softlabels(fold)

    def run(self,fold):
        print('training size: ', len(self.X_train) , ' validation size: ', len(self.X_validation))
        results=None
        y_pred={}
        models={}

        if self.args.type == 'maxent':
            model = MaxentClassifier.train(list( zip(self.X_train,self.Y_train)), max_iter=200)
            test_featuresets =  self.X_validation
            y_pred = model.classify_many(test_featuresets)
            y_pred=pd.DataFrame(y_pred)

        elif self.args.type == 'nn':

            if args.augment:
                for a in self.group_augment:
                    self.X_train.extend(list(self.group_augment[a]['text'].values))
                    self.Y_train.extend(list(self.group_augment[a]['groupID'].values))

            if self.args.oversample:
                self.oversample()
            elif self.args.downsample > 0:
                self.downsample(self.args.downsample)

            models = BERT(  self.X_train, self.X_validation,self.Y_train, self.Y_train_softlabels if self.args.distill else None,
                             self.Y_validation, self.args,
                             self.groups,
                             self.X_train_teacher if self.args.distill else None ,
                             self.Y_train_teacher if self.args.distill else None )
            self.retry = True
            results = models.train(fold,self.args.distill)

        else:
            if(self.args.type=='svm'):
                models = svm.SVC(kernel='linear',probability=True,decision_function_shape='ovr')# LinearSVC()
            elif self.args.type == 'randomforest':
                models = RandomForestClassifier( random_state=0, n_estimators=10,max_features=None)
            elif self.args.type == 'decisiontree':
                models = DecisionTreeClassifier( random_state=43)
            elif self.args.type == 'gbm':
                models = GradientBoostingClassifier( random_state=0)

            print('training ... fold_{}_model_{}'.format(fold, self.args.type))
            models.fit(self.features, self.Y_train)

        if self.args.type != 'nn':
            y_pred = models.predict(self.features_validation)
            y_prob = models.predict_proba(self.features_validation)
            pickle.dump([self.Y_validation,   y_pred, [max(p) for p in y_prob] ], open('fold_{}_preds.pickle'.format(fold), 'wb'))
            pickle.dump([self.Y_validation,   y_pred,   y_prob ], open('results_fold_{}_preds.pickle'.format(fold), 'wb'))

            lbs = list(set(self.Y_validation))
            lbs.sort()
            print ('*'*20, 'fold {}'.format(fold) ,'*'*20)
            print( classification_report(self.Y_validation, y_pred, labels=groups))
            results = (classification_report(self.Y_validation, y_pred, labels=groups, output_dict=False),
                       classification_report(self.Y_validation, y_pred, labels=groups, output_dict=True))


        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--testrun', action='store_true', default=False)
    parser.add_argument('--ng', default=2, type=int )
    parser.add_argument('--type', default='svm')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--nocoded', action='store_true', default=False)
    parser.add_argument('--notes', action='store_true', default=True)
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--smote',  default='SMOTEENN')
    parser.add_argument('--downsample', type=int, default=0)
    parser.add_argument('--oversample', action='store_true', default=False)
    parser.add_argument('--mt', action='store_true', default=False)
    parser.add_argument('--p',type=int, default=5)
    parser.add_argument('--norun',action='store_true', default=False)
    parser.add_argument('--nn_teacher_model', default='bert-base-uncased')
    parser.add_argument('--nn_model',default='bert-base-uncased')
    parser.add_argument('--earlystop',action='store_true', default=False)
    parser.add_argument('--patience', default=5, type=int )
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--augment_size', type=int, default=200)
    parser.add_argument('--augment_per_instance', type=int, default=30)
    parser.add_argument('--augment_fold', type=int, default=-1)
    parser.add_argument('--softlabels',  action='store_true', default=False)
    parser.add_argument('--distill',  action='store_true', default=False)
    parser.add_argument('--distill_weight', type=float, default=0)

    args = parser.parse_args()

    if platform == "darwin" and (args.type == 'nn' or args.notes == True):
        args.testrun = True
        # args.augment=True
        # args.softlabels=True
        # args.distill = True
        args.oversample=True
        args.nn_teacher_model = './saved_model'

    print(args)

    results=[]

    if args.augment:
        au = auto(args)
        for fold in range(5):
            if args.fold != None and fold != args.fold: continue
            au.prep(fold)
            au.cv(fold)

    elif args.softlabels:
        au = auto(args)
        for fold in range(5):
            if args.fold != None and fold != args.fold: continue
            au.prep(fold)
            au.cv(fold)
            au.generate_softlabels(fold)
    else:

        def cv_fold(fold):
            au = auto(args)
            au.prep(fold)
            au.cv(fold)
            if not args.norun:
                results = au.run(fold)

                return results
            else:
                return None


        if args.type != 'nn':
            if args.fold != None:
                folds = [args.fold]
            else:
                folds = [[i] for i in range(5)]

            if args.mt :
                it=iter(folds)
                from multiprocessing import Pool
                with Pool(processes=args.p if platform != "darwin" else 1) as pool:
                    results=pool.starmap(cv_fold, it)
            else:
                for fold in range(5):
                    if args.fold != None and fold != args.fold: continue
                    results.append(cv_fold(fold))

        else:
            for fold in range(5):
                if args.fold != None and fold != args.fold: continue
                results.append(cv_fold(fold))

        pickle.dump(results, open('results.pkl', 'wb'))

        if args.fold != None:
            print('*' * 20, 'fold {}'.format(fold), '*' * 20)
            if results[0] is not None:
                print(results[0][0])
                results[0] = results[0][1]

        else:
            for fold in range(5):
                print('*' * 20, 'fold {}'.format(fold), '*' * 20)
                if results[fold] is not None:
                    print(results[fold][0])
                    results[fold] = results[fold][1]

        if results.__len__() > 0:
            print('p {:.2f} r {:.2f} f1 {:.2f}'.format(np.mean([r['macro avg']['precision'] for r in results]),
                                                       np.mean([r['macro avg']['recall'] for r in results]),
                                                       np.mean([r['macro avg']['f1-score'] for r in results])))
