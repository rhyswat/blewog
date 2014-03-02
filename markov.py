#!/usr/bin/python2

# Markov chain stuff

import bisect
import collections
import itertools
import functools
import random

class State :
    def __init__(self, *transitions) :
        self.transitions = transitions

    def __str__(self) :
        return str(self.transitions)

    def normalise(self) :
        n = float(sum(self.transitions))
        self.transitions = tuple(t/n for t in self.transitions)
        
    def update(self) :
        pass

    def __getitem__(self, i) :
        return self.transitions[i]

    def vector(self) :
        return self.transitions[:]

KEY = {'W':0, 'D':1, 'L':2}
class Team :
    def __init__(self, name) :
        self.name = name
        self.last_result = 'D'
        self.wins = 0
        self.draws = 0
        self.defeats = 0
        # Each state is {p(win), p(draw), p(lose)}
        self.transitions = {'W':State(1,1,1), 'D':State(1,1,1), 'L':State(1,1,1)}
        for t in self.transitions.values() :
            t.normalise()

    def details(self) :
        print self.name
        print '   win-transitions =',self.transitions['W']
        print '  draw-transitions =',self.transitions['D']
        print '  lose-transitions =',self.transitions['L']
        print '       last-result =', self.last_result

    def get_transitions(self) :
        return self.transitions[self.last_result]

    def set_transitions(self, **tx) :
        for k,v in tx.iteritems() :
            if k in self.transitions :
                s = State(*v)
                s.normalise()
                self.transitions[k] = s

    def win(self) :
        self.wins += 1
        self.last_result = 'W'

    def draw(self) :
        self.draws += 1
        self.last_result = 'D'

    def lose(self) :
        self.defeats += 1
        self.last_result = 'L'
    
    def points(self) :
        return 3*self.wins + self.draws

    def __str__(self) :
        return str(self.name)

    def __repr__(self) :
        return repr(self.name)

    def __hash__(self) :
        return hash(self.name)

    def __eq__(self, other) :
        return self.name == other.name
            
def league_order(a, b) :
    return cmp((a.points(), a.wins, a.draws, a.name), (b.points(), b.wins, b.draws, b.name))

def print_league(unsorted_teams) :
    teams = sorted(unsorted_teams, cmp=league_order, reverse=True)
    longest = max(len(t.name) for t in teams)
    fmt = '{:%s} |  W |  D |  L | Pts |' % longest
    s = fmt.format('Team')
    print ('=' * len(s))
    print s
    print ('=' * len(s))
    fmt = '{:%s} | {:2d} | {:2d} | {:2d} | {:3d} |' % longest
    for t in teams :
        print fmt.format(t.name, t.wins, t.draws, t.defeats, t.points())
    print ('=' * len(s))
    
def make_league() :
    # need an even number
    names = ['Chelsea',
             'Liverpool',
             'Arsenal',
             'Manchester City',
             'Tottenham',
             'Everton',
             'Manchester United',
             'Newcastle United',
             'Southampton',
             'West Ham',
             'Hull City',
             'Stoke City',
             'Swansea City',
             'Aston Villa',
             'Norwich City',
             'Crystal Palace',
             'West Brom',
             'Sunderland',
             'Cardiff City',
             'Fulham']

    teams = {}
    for name in names :
        teams[name] = Team(name)

    # vary the teams
    teams['Cardiff City'].set_transitions(W=(10,1,1))
    teams['Everton'].set_transitions(D=(10,10,1))
    
    
    return teams.values()

def make_schedule(teams) :
    n = len(teams)
    n2 = n/2
    homes, aways = teams[:n2], teams[n2:]

    # round-robin scheduling using homes[0] as the pivot,
    # this generates the first half of the season only,
    # so we simultaneously construct the second half by mirroring.
    first = []
    second = []
    for i in xrange(n-1) :
        first.append([(homes[j], aways[j]) for j in xrange(n2)])
        second.append([(aways[j], homes[j]) for j in xrange(n2)])
        x = homes.pop(-1)
        y = aways.pop(0)
        homes.insert(1, y)
        aways.append(x)
        
    # full schedule = first half + reversed of second half
    return first + list(reversed(second))


def play(home, away) :
    # arrange the probabilities in a wheel of 6 slices: 
    # 3 home transitions then 3 away transitions.
    # then normalise and compute a pdf
    p = home.get_transitions().vector() + away.get_transitions().vector()
    n = float(sum(p))
    p = [t/n for t in p]
    pdf = [0] * len(p)
    pdf[0] = p[0]
    for i in range(1, len(p)) :
        pdf[i] = pdf[i-1] + p[i]
    outcome = random.uniform(0, 1)
    index = bisect.bisect_left(pdf, outcome)
    if index == 0 or index == 5 :
        home.win()
        away.lose()
    elif index == 2 or index == 3:
        home.lose()
        away.win()
    else :
        home.draw()
        away.draw()

if __name__ == '__main__' :
    random.seed(2435L)
    league = make_league()
    schedule = make_schedule(league)
    r = 1
    for rnd in schedule :
        for match in rnd :            
            play(*match)
            # then update probabilities....
        r += 1
    print ''
    print 'Season over:'
    print_league(league)

    #print ''
    #print 'Final transitions:'
    #for team in league :
    #    team.details()

