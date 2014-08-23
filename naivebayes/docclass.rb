#!/usr/local/bin/ruby
# -*- coding: utf-8 -*-

module Classifier

  # ----------------------------------------------------------------------------
  # demo methods

  def self.test_bayes()
    b = Classifier::NaiveBayes.new
    train(b)
    
    [['quick rabbit', :good], ['quick rabbit', :bad]].each do |(item, c)|
      p = b.prob(c, item)
      puts "P(#{c}|'#{item}') = #{p}"
    end
    
    b.setThreshold(:bad, 3.0)
    c = b.classify('quick money')
    puts "'quick money' is #{c}"
    
    10.times { |n| train(b) }
    puts "After re-training...."
    c = b.classify('quick money')
    puts "'quick money' is #{c}"
  end

  # ----------------------------------------------------------------------------
  # bootstrap training
  def self.train(cl)
    cl.train('Nobody owns the water', :good)
    cl.train('the quick rabbit jumps fences', :good)
    cl.train('buy pharmaceuticals now', :bad)
    cl.train('make quick money at online casino', :bad)
    cl.train('the quick brown fox jumps', :good)
  end

  # Naive Bayesian classifier a la ch 6 of Segaran
  class Classifier
 
    # Constructor - pass in a callable which generates a feature vector
    def initialize(getfeatures=nil, filename=nil) 
      # counts of features per category
      @fc = Hash.new(Hash.new(0))
       # input items per category
      @cc = Hash.new(0)
       # feature extraction function
      getfeatures = :split.to_proc if getfeatures.nil?
      @getfeatures = getfeatures
      # classification thresholds
      @thresholds = Hash.new(1.0)
    end
    
    # Utilities ------------------------------------------------------------------
    # Word splitter
    def split(line)
      return line.split("\\W*").uniq
    end

    def incf(feature, cat)
      @fc[feature] = Hash.new(0) unless @fc.include?(feature)
      @fc[feature][cat] = @fc[feature][cat] + 1
    end
    
    def incc(cat)
      @cc[cat] = @cc[cat] + 1
    end
    
    # the number of times this category has been associated with this feature
    def fcount(f, cat)
      return @fc[f][cat].to_f if (@fc.include?(f) && @fc[f].include?(cat))
      return 0.0
    end
    
    def catcount(cat)
      return @cc[cat].to_f
    end
    
    def totalcount()
      @cc.inject(0) { |memo, (k,v)| memo + v }
    end
    
    def categories()
      @cc.keys
    end
    
    # Training -----------------------------------------------------------------
    
    # Pass in a feature to break into a vector, and a label e.g. good/bad
    def train(item, label)
      featureVector = @getfeatures.call(item)
      featureVector.each {|f| incf(f, label) }
      incc(label)
    end
    
    # Probabilities ------------------------------------------------------------

    # P(F|C) -- feature given a class
    def fprob(f, c)
      return 0.0 if catcount(c).zero?
      return fcount(f,c)/catcount(c)
    end

    # weighted probability based on prf(f,c) = P(F|C) to get around zero counts
    def weightedprob(f, c, w=1.0, assumed=0.5)
      p = fprob(f, c)
      total = categories().inject(0) { |memo, cat| memo + fcount(f, cat) }
      return (w * assumed + total * p)/(w + total)
    end

    # P(C|Item) -- TO BE OVERRIDDEN
    def prob(c, item)
      return 0.0
    end

    # Classifying --------------------------------------------------------------
    
    def setThreshold(c, t)
      @thresholds[c] = t
    end

    def classify(item, default=:unknown)
      # class with the highest probability
      probs = {}
      max = 0
      best = default
      categories.each do |c|
        probs[c] = prob(c, item)
        if probs[c] > max
          max = probs[c]
          best = c
        end
      end

      max = probs.delete best
      probs.each do |c, p|
        return default if p*@thresholds[best] > max
      end
      return best
    end

  end

  class NaiveBayes < Classifier

    # P(Item, Class) = P(F1,F2,...Fn|Class) using the naive independence assumption
    # where [F1, F2, ..., Fn] is the featutre vector generated from an input item.
    def likelihood(item, c)
      @getfeatures.call(item).inject(1) { |memo, f| memo * weightedprob(f, c) }
    end

    # P(Class) -- the prior probability of finding this class.
    # Crucially, this changes with every training update.
    # Equal to the frequency of this label amongst all labels seen so far.
    def prior(c)
      catcount(c)/totalcount()
    end
    
    # Posterior     = Prior * Likelihood / Evidene
    # P(Class|Item) = P(Class) * P(Item|Class) / Evidence
    # This computes the numerator for the posterior because Evidence is constant
    # Note that Sum(Posteriors/all classes) = 1 including at the beginning
    # so Evidence = Sum(P(Class) * product(P(Feature|Class) over all features) over all classes)
    def prob(c, item)
      return prior(c)*likelihood(item, c)
    end
    
  end
  
end # module Classifier

# Test code
if __FILE__ == $0
  include Classifier
  puts 'Testing crappy spam filter'
  Classifier::test_bayes
  puts 'Done'
end
