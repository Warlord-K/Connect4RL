import numpy as np


class Connect4():
    def __init__(self, board_size = (6,7)):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)
        self.turn = 1
        self.done = False
        self.winner = 0
        self.state = 0


    def encode(self,action):
        turn = self.turn if self.turn != -1 else 2
        self.state = self.state + turn* (3**action)
    
    
    def step(self,col):
        reward = -1
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            action = row*self.board_size[0] + col
            self.encode(action)
            self.board[row][col]= self.turn
            self.done = self.winning_move()
            self.winner = self.turn if self.done else 0
            self.turn *= -1
            if self.done:
                reward = 100
            if self.is_board_full():
                self.done = True
                reward = 0
            return self.state,reward,self.done,self.winner
            
        else:
            reward = -10
            return self.state,reward,self.done,self.winner
        

    def is_valid_location(self,col):
        #if this condition is true we will let the use drop self.turn here.
        #if not true that means the col is not vacant
        return self.board[5][col]==0

    def get_next_open_row(self,col):
        for r in range(self.board_size[0]):
            if self.board[r][col]==0:
                return r

    def winning_move(self):
        # Check horizontal locations for win
        for c in range(self.board_size[1]-3):
            for r in range(self.board_size[0]):
                if self.board[r][c] == self.turn and self.board[r][c+1] == self.turn and self.board[r][c+2] == self.turn and self.board[r][c+3] == self.turn:
                    return True
    
        # Check vertical locations for win
        for c in range(self.board_size[1]):
            for r in range(self.board_size[0]-3):
                if self.board[r][c] == self.turn and self.board[r+1][c] == self.turn and self.board[r+2][c] == self.turn and self.board[r+3][c] == self.turn:
                    return True
    
        # Check positively sloped diaganols
        for c in range(self.board_size[1]-3):
            for r in range(self.board_size[0]-3):
                if self.board[r][c] == self.turn and self.board[r+1][c+1] == self.turn and self.board[r+2][c+2] == self.turn and self.board[r+3][c+3] == self.turn:
                    
                    return True
    
        # Check negatively sloped diaganols
        for c in range(self.board_size[1]-3):
            for r in range(3, self.board_size[0]):
                if self.board[r][c] == self.turn and self.board[r-1][c+1] == self.turn and self.board[r-2][c+2] == self.turn and self.board[r-3][c+3] == self.turn:
                    
                    return True
        
            
        return False

    def is_board_full(self):
        for i in range(7):
            if self.is_valid_location(i):
                return False
        return True
    def render(self):
        print(np.flip(self.board,0))
        

            
    
    
