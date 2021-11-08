#from monte_carlo.basis.dependencies import *
#from monte_carlo.basis.environment import *
from monte_carlo.basis.dependencies import *
from monte_carlo.basis.environment import *
class MC_Simple_Extractor():

    def __init__(self):
        self.n_features = 25

    def get_num_features(self):
        return self.n_features
    
    def get_features (self, state, action):
        
        f = {}
        # notice that the legal actions were previously selected.
        dx, dy = Agent.get_direction(action)

        # positions(x,y): width, height 0-1, player 2-3, hunter 4-5, target 6-7
        w , h  = state[0][0], state[0][1]
        pX, pY = state[0][2] + dx, state[0][3] + dy
        if w == 0:
            w=0.1**(-7)
        if h == 0:
            h=0.1**(-7)

        """
        here, we predict what will be the closest future position the hunter might be.
        """
        hX, hY = state[0][4], state[0][5]

        dist_x = pX - hX 
        if dist_x > 0: 
            hX += 1
        elif dist_x < 0: 
            hX -= 1
        
        dist_y = (pY - hY)
        if dist_y > 0: 
            hY += 1
        elif dist_y < 0: 
            hY -= 1           
         
        tX, tY = state[0][6], state[0][7]

        prev_pX, prev_pY = state[0][2], state[0][3]
        prev_hX, prev_hY = state[0][4], state[0][5]
        prev_tX, prev_tY = state[1][6], state[1][7]

        # 1
        '''
        f['bias'] = 1.0
        '''
        # 2
        f['won'] = int(pX == tX and pY == tY)
        #
        f['died'] = int(pX == hX and pY == hY)
        # 4
       
        f['player-pos-x'] = float(pX/w)
        # 
        f['player-pos-y'] = float(pY/h)
        # 6
        f['dist-to-hunter-X'] = float((hX - pX)/w)
        # 
        f['dist-to-hunter-Y'] = float((hY - pY)/h)
        # 8
        f['dist-to-target-X'] = float((tX - pX)/w)
        # 
        f['dist-to-target-Y'] = float((tY - pY)/h)
        # 10
        f['action'] = float(action/3)
        #
        f['pos-hunter-X'] = float(hX/w)
        # 12
        f['pos-hunter-Y'] = float(hY/h)
        #
        f['pos-target-X'] = float(tX/w)
        # 14
        f['pos-target-Y'] = float(tY/h)
        #
        '''
        a = float((hX - pX)/w)
        b = float((hY - pY)/h)
        f['2-dist-to-hunter'] = 2 - (abs(a) + abs(b))

        a = float((tX - pX)/w)
        b = float((tY - pY)/h)
        # 16
        f['2-dist-to-target'] = 2 - (abs(a) + abs(b))
        
        
        f['prev_dist-to-hunter-X'] = float((prev_hX - prev_pX)/w)
        f['prev_dist-to-hunter-Y'] = float((prev_hY - prev_pY)/h)
        f['prev_dist-to-target-X'] = float((prev_tX - prev_pX)/w)
        f['prev_dist-to-target-Y'] = float((prev_tY - prev_pY)/h)
        #
        
        f['2-prev_dist-to-hunter'] = 2 - (abs(f['prev_dist-to-hunter-X']) + abs(f['prev_dist-to-hunter-Y']))
        f['2-prev_dist-to-target'] = 2 - (abs(f['prev_dist-to-target-X']) + abs(f['prev_dist-to-target-Y']))
        #
        
        f['diff-distances-hunter'] = f['2-dist-to-hunter'] - f['2-prev_dist-to-hunter']
        # 24
        f['diff-distances-target'] = f['2-dist-to-target'] - f['2-prev_dist-to-target']
        #
        
        f['hunter-1-step-away'] = abs(pX - hX) + abs(pY - hY) <= 1
        '''
        return f 
    
    def ini(self):
        f = {}
        '''
        f['bias'] = 1.0
        # 2
        f['won'] = np.random.random()
        #
        f['died'] = np.random.random()
        f['player-pos-x'] = np.random.random()
        f['player-pos-y'] = np.random.random()
        f['dist-to-hunter-X'] = np.random.random()
        # 
        f['dist-to-hunter-Y'] = np.random.random()
        # 8
        f['dist-to-target-X'] = np.random.random()
        # 
        f['dist-to-target-Y'] = np.random.random()
        # 10
        f['action'] = np.random.random()
        #
        f['pos-hunter-X'] = np.random.random()
        # 12
        f['pos-hunter-Y'] = np.random.random()
        #
        f['pos-target-X'] =np.random.random()
        # 14
        f['pos-target-Y'] = np.random.random()
        # 
        '''
        f['2-dist-to-hunter'] = np.random.random()
        # 16
        f['2-dist-to-target'] = np.random.random()
        #
        '''
        f['prev_dist-to-hunter-X'] = np.random.random()
        f['prev_dist-to-hunter-Y'] = np.random.random()
        f['prev_dist-to-target-X'] = np.random.random()
        f['prev_dist-to-target-Y'] = np.random.random()
        #
        
        f['2-prev_dist-to-hunter'] = np.random.random()
        f['2-prev_dist-to-target'] = np.random.random()
        #
        
        f['diff-distances-hunter'] = np.random.random()
        # 24
        f['diff-distances-target'] = np.random.random()
        # 
        f['hunter-1-step-away'] = np.random.random()
        '''
        return f