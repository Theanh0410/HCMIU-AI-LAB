import pandas as pd
import numpy as np
import sys

class NaiveBayesFilter:
    def __init__(self):
        self.data = []
        self.vocabulary = []  # returns tuple of unique words
        self.p_spam = 0  # Probability of Spam
        self.p_ham = 0  # Probability of Ham
        # Initiate parameters
        self.parameters_spam = {unique_word: 0 for unique_word in self.vocabulary}
        self.parameters_ham = {unique_word: 0 for unique_word in self.vocabulary}

    def fit(self, X, y):
        "*** YOUR CODE HERE ***"
        #Create DataFrame
        self.data = pd.DataFrame({'SMS': X, 'Label': y})
        
        #Calculate prior probabilities
        spam_mes = self.data[self.data['Label'] == 'spam']
        ham_mes = self.data[self.data['Label'] == 'ham']
        
        #Equations 1.3 and 1.4
        self.p_spam = len(spam_mes)/len(self.data)
        self.p_ham = len(ham_mes)/len(self.data)
        
        #Build vocabulary and calculate word counts
        self.vocabulary = set()
        for mes in self.data['SMS']:
            for word in mes:          
                self.vocabulary.add(word)
                
        print("vocab count:", len(self.vocabulary))
                        
        spam_word_counts = {word: 0 for word in self.vocabulary}
        ham_word_counts = {word: 0 for word in self.vocabulary}
        
        for mes in spam_mes['SMS']:
            for word in mes:
                spam_word_counts[word] += 1
                
        for mes in ham_mes['SMS']:
            for word in mes:
                ham_word_counts[word] += 1
                
        #Apply Laplace smoothing to calculate conditional probabilities
        spam_total_words = sum(spam_word_counts.values())
        ham_total_words = sum(ham_word_counts.values())
        
        self.parameters_spam = {
            word: (spam_word_counts[word] + 1)/(spam_total_words + len(self.vocabulary))
            for word in self.vocabulary
        }
        
        self.parameters_ham = {
            word: (ham_word_counts[word] + 1)/(ham_total_words + len(self.vocabulary))
            for word in self.vocabulary
        }
        
        print('parameters_spam: ')
        print(dict(list(self.parameters_spam.items())[:10]))
        print('parameters_ham: ')
        print(dict(list(self.parameters_ham.items())[:10]))
        
        return self.data

    def predict(self, X):
        prob = []
        "*** YOUR CODE HERE ***"        
        for p_ham, p_spam in self.predict_proba(X):
            if p_spam > p_ham:
                prob.append('spam')
            else:
                prob.append('ham')
                
        return prob

    def predict_proba(self, X):
        proba = []
        "*** YOUR CODE HERE ***"
        for mes in X:
            p_spam_given_mes = self.p_spam
            p_ham_given_mes = self.p_ham
            
            #Multiply the likelihood of each word
            for word in mes:
                if word in self.parameters_spam:
                    p_spam_given_mes *= self.parameters_spam[word]
                if word in self.parameters_ham:
                    p_ham_given_mes *= self.parameters_ham[word]
                    
            proba.append([p_spam_given_mes, p_ham_given_mes])
        
        return proba

    def score(self, true_labels, predict_labels):
        recall = 0
        "*** YOUR CODE HERE ***"
        true_pos = 0
        false_neg = 0
        
        #Calculate true-positive and false-negative
        for true, pred in zip(true_labels, predict_labels):
            if true == 'spam' and pred == 'spam':
                true_pos += 1
            if true == 'spam' and pred == 'ham':
                false_neg += 1

        #Apply Recall-precision curve
        if (true_pos + false_neg) > 0:
            recall = true_pos/(true_pos + false_neg)
        else:
            recall = 0
        
        return recall