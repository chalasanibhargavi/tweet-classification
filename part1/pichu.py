#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:36:49 2017

@author: Ankit
"""
import sys
import numpy as np

class Chess_Graph:
    
    def __init__(self, start_boardObj, current_player, max_depth):
        self.start_boardObj = start_boardObj
        self.current_player = current_player
        self.start_player = current_player
        self.max_depth = max_depth
        self.curr_depth = max_depth
        self.initial_depth = 0
        self.best_board = None
        self.piece_weights = {'k': 100, 'p': 1, 'b': 3, 'n': 3, 'r': 5, 'q': 20}
        self.loc_weights = {'k': 0, 'p': 1, 'b': 3, 'n': 3, 'r': 5, 'q': 20}
        pass
    
    def swap_player(self):
        if self.current_player == 'w':
                self.current_player = 'b'
        elif self.current_player == 'b':
                self.current_player = 'w'
        
    def minimax(self, curr_boardObj, depth):
        self.current_player = self.start_player
        self.initial_depth = depth
        best_value = self.max_val(curr_boardObj, depth)

        return best_value
        pass
    
    def max_val(self, curr_boardObj, depth):
        if (self.initial_depth - depth) % 2 != 0:
            self.swap_player()
        
        if depth == 0:
            return curr_boardObj.value
        else:
            max_value = -float(1000000)
            list_succ = self.successors(curr_boardObj)
            if list_succ != None:
                for temp_board in list_succ:
                    temp_value = self.min_val(temp_board, depth - 1)
                    if temp_value >= max_value:
                        max_value = temp_value
                        if depth == self.initial_depth:
                            self.best_board = temp_board
                
            return max_value
        pass
            
    def min_val(self, curr_boardObj, depth):
        if (self.initial_depth - depth) % 2 != 0:
            self.swap_player()
        
        if depth == 0:
            return curr_boardObj.value
        else:
            min_value = float(1000000)
            list_succ = self.successors(curr_boardObj)
            if list_succ != None:
                for temp_board in list_succ:
                    temp_value = self.max_val(temp_board, depth - 1)
                    if temp_value <= min_value:
                        min_value = temp_value
                    
            return min_value
        pass
    
    def alpha_beta(self, curr_boardObj, depth):
        self.current_player = self.start_player
        self.initial_depth = depth
        best_value = self.max_ab(curr_boardObj, depth, -float(1000000), float(1000000))

        return best_value
        pass
    
    def max_ab(self, curr_boardObj, depth, alpha, beta):
        if (self.initial_depth - depth) % 2 != 0:
            self.swap_player()
                
        if depth == 0:
            return curr_boardObj.value
        else:
            list_succ = self.successors(curr_boardObj)
            if list_succ != None:
                for temp_board in list_succ:
                    temp_alpha = self.min_ab(temp_board, depth - 1, alpha, beta)
                    if temp_alpha >= alpha:
                        alpha = temp_alpha
                        if depth == self.initial_depth:
                            self.best_board = temp_board
                        
                    if alpha >= beta:
                        return alpha
            return alpha
        
    def min_ab(self, curr_boardObj, depth, alpha, beta):
        if (self.initial_depth - depth) % 2 != 0:
            self.swap_player()
            
        if depth == 0:
            return curr_boardObj.value
        else:
            list_succ = self.successors(curr_boardObj)
            if list_succ != None:
                for temp_board in list_succ:
                    temp_beta = self.max_ab(temp_board, depth - 1, alpha, beta)
                    if temp_beta <= beta:
                        beta = temp_beta
                    if alpha >= beta:
                        return beta
                    
            return beta
    
    def calc_heuristic(self, board):
        final_score = 0.0
        
        #Heuristic 1: sum of piece counts
        #Using the dictionary 'self.piece_weights' we calculate the sum of piece weights for both white and black pieces.
        #If the board has a Kingfisher piece then it should get more weight compared to a board with no Kingfisher piece.
        #Then Quetzal gets more weight compared to other pieces as it can move in all directions.
        #Then Robin gets more preference then the remaining pieces as it can move freely horizontally and vertically, as so on.
        #The weights assigned are random numbers and some ideas have been taken from the heuristics mentioned in the following:
        #https://github.com/lamesjim/Chess-AI
        
        #Heuristic 2: location of the pieces on the board
        #Using the dictionary 'self.loc_weights' we calculate the sum of the piece location on the board for both black and white pieces.
        #Pieces that are located in the center 2*2 matrix of the 8*8 board have more mobility compared to the pieces on the remaining locations.
        #Pieces on row3 and row6 are also more flexible then pieces on the row1, row2, row7, and row8.
        
        score_heur1_b = 0.0
        score_heur1_w = 0.0
        
        score_heur2_b = 0.0
        score_heur2_w = 0.0
        
        for r in range(0, 8):
            for c in range(0, 8):
                piece = board[r][c]
                #Turn - BLACK
                if piece.islower():
                    score_heur1_b += self.piece_weights[piece]
                    if r in [3, 4] and c in [3, 4]:
                        score_heur2_b += 2 * self.loc_weights[piece]    
                        
                    if r in [2, 5] and c in [2, 3, 4, 5]:
                        score_heur2_b += 0.5 * self.loc_weights[piece]
                        
                    if r in [3, 4] and c in [2, 5]:
                        score_heur2_b += 1 * self.loc_weights[piece]
                        
                #Turn - WHITE
                elif piece.isupper():
                    score_heur1_w += self.piece_weights[piece.lower()]
                    if r in [3, 4] and c in [3, 4]:
                        score_heur2_w += 2 * self.loc_weights[piece.lower()]    
                        
                    if r in [2, 5] and c in [2, 3, 4, 5]:
                        score_heur2_w += 0.5 * self.loc_weights[piece.lower()]
                        
                    if r in [3, 4] and c in [2, 5]:
                        score_heur2_w += 1 * self.loc_weights[piece.lower()]
                
        if self.current_player == 'b':
            score_heur1 = score_heur1_b - score_heur1_w
            score_heur2 = score_heur2_b - score_heur2_w
        elif self.current_player == 'w':
            score_heur1 = score_heur1_w - score_heur1_b
            score_heur2 = score_heur2_w - score_heur2_b
            
        final_score += score_heur1 * 4
        final_score += score_heur2 * 2
        
        return final_score
        
    #Used to create a board object based on the Heuristic value calculated from calc_heuristic() method.
    def makeBoardObj(self, board):
        heurisitic_value = self.calc_heuristic(board)
        boardObj = Chess_Board(board, heurisitic_value)
        return boardObj
        
    #Used to swap two pieces on the board. ele1 is moved from position (x1, y1) to (x2, y2) on the board.
    def swap_blocks(self, curr_board, x1, y1, ele1, x2, y2, ele2):
        sample = curr_board.copy()
        sample[x1][y1] = ele1
        sample[x2][y2] = ele2
        
        boardObj = self.makeBoardObj(sample)
        return boardObj
        
    #To find out the possible successors for all the parakeets of the current player.
    def move_parakeet(self, curr_board, positions):
        temp = []
        temp_board = []

        if self.current_player == 'w':
            key = 'P'
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'p'
            opp_list = pieces_names.keys()
        
        for pos in positions:
            r, c = pos[0], pos[1]

            if self.current_player == 'w':
                #move straight
                if r <= 6 and curr_board[r + 1][c] == '.':
                    if r == 6:
                        #Pawn becomes a Queen on reaching the end of the board
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c, 'Q')
                    else:
                        #move the pawn to the next row and same column
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c, key)
                    temp.append(temp_board)
                
                #move straight two places on its first move
                if r == 1 and curr_board[r + 1][c] == '.' and curr_board[r + 2][c] == '.':
                    temp_board = self.swap_blocks(curr_board, r, c, '.', r + 2, c, key)
                    temp.append(temp_board)
                
                #kill an opponent piece on lower left diagonal
                if r <= 6 and c >= 1:
                    if curr_board[r + 1][c - 1] in opp_list:
                        if r == 6:
                            #Pawn becomes a Queen on reaching the end of the board
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c - 1, 'Q')
                        else:
                            #move the pawn to the next row and c - 1 column
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c - 1, key)
                        temp.append(temp_board)
                        
                #kill an opponent piece on lower right diagonal
                if r <= 6 and c <= 6:
                    if curr_board[r + 1][c + 1] in opp_list:
                        if r == 6:
                            #Pawn becomes a Queen on reaching the end of the board
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c + 1, 'Q')
                        else:
                            #move the pawn to the next row and c + 1 column
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c + 1, key)
                        temp.append(temp_board)
                        
            else:
                #move straight
                if r >= 1 and curr_board[r - 1][c] == '.':
                    if r == 1:
                        #Pawn becomes a Queen on reaching the end of the board
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c, 'q')
                    else:
                        #move the pawn to the next row and same column
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c, key)
                    temp.append(temp_board)
                
                #move straight two places on its first move
                if r == 6 and curr_board[r - 1][c] == '.' and curr_board[r - 2][c] == '.':
                    temp_board = self.swap_blocks(curr_board, r, c, '.', r - 2, c, key)
                    temp.append(temp_board)
                
                #kill an opponent piece on upper left diagonal
                if r >= 1 and c >= 1:
                    if curr_board[r - 1][c - 1] in opp_list:
                        if r == 1:
                            #Pawn becomes a Queen on reaching the end of the board
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c - 1, 'q')
                        else:
                            #move the pawn to the next row and c - 1 column
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c - 1, key)
                        temp.append(temp_board)
                        
                #kill an opponent piece on lower right diagonal
                if r >= 1 and c <= 6:
                    if curr_board[r - 1][c + 1] in opp_list:
                        if r == 6:
                            #Pawn becomes a Queen on reaching the end of the board
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c + 1, 'q')
                        else:
                            #move the pawn to the next row and c + 1 column
                            temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c + 1, key)
                        temp.append(temp_board)
            
        return temp
        pass
    
    #To find out the possible successors for all the nighthawks of the current player.
    def move_nighthawk(self, curr_board, positions):
        temp = []
        temp_board = []
        
        if self.current_player == 'w':
            key = 'N'
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'n'
            opp_list = pieces_names.keys()
            
        for pos in positions:
            x, y = pos[0], pos[1]

            if y + 1 < 8 and x - 2 >= 0 and (curr_board[x - 2][y + 1] == "." or curr_board[x - 2][y + 1] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x - 2, y + 1, key)
                temp.append(temp_board)

            if y + 1 < 8 and x + 2 < 8 and (curr_board[x + 2][y + 1] == "." or curr_board[x + 2][y + 1] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x + 2, y + 1, key)
                temp.append(temp_board)
                
            if y + 2 < 8 and x - 1 >= 0 and (curr_board[x - 1][y + 2] == "." or curr_board[x - 1][y + 2] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x - 1, y + 2, key)
                temp.append(temp_board)

            if y + 2 < 8 and x + 1 < 8 and (curr_board[x + 1][y + 2] == "." or curr_board[x + 1][y + 2] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x + 1, y + 2, key)
                temp.append(temp_board)

            if y - 1 >= 0 and x - 2 >= 0 and (curr_board[x - 2][y - 1] == "." or curr_board[x - 2][y - 1] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x - 2, y - 1, key)
                temp.append(temp_board)

            if x + 2 < 8 and y - 1 >= 0 and (curr_board[x + 2][y - 1] == "." or curr_board[x + 2][y - 1] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x + 2, y - 1, key)
                temp.append(temp_board)

            if x - 1 >= 0 and y - 2 >= 0 and (curr_board[x - 1][y - 2] == "." or curr_board[x - 1][y - 2] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x - 1, y - 2, key)
                temp.append(temp_board)

            if x + 1 < 8 and y - 2 >= 0 and (curr_board[x + 1][y - 2] == "." or curr_board[x + 1][y - 2] in opp_list):
                temp_board = self.swap_blocks(curr_board, x, y, '.', x + 1, y - 2, key)
                temp.append(temp_board)
        
        return temp
        pass
    
    #To find out the possible successors for all the robins of the current player.
    def move_robin(self, curr_board, positions):
        temp = []
        temp_board = []
        
        if self.current_player == 'w':
            key = 'R'
            same_list = pieces_names.keys()
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'r'
            same_list = map(lambda x: x.lower(), pieces_names.keys())
            opp_list = pieces_names.keys()
        
        for pos in positions:
            r, c = pos[0], pos[1]
            
            for i in range(r - 1, -1, -1):
                if curr_board[i][c] in same_list:
                    break;
                else:
                    if curr_board[i][c] == '.' or curr_board[i][c] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', i, c, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[i][c] in opp_list:
                            break
                        
            for i in range(r + 1, 8):
                if curr_board[i][c] in same_list:
                    break;
                else:
                    if curr_board[i][c] == '.' or curr_board[i][c] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', i, c, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[i][c] in opp_list:
                            break
                    
            for j in range(c - 1, -1, -1):
                if curr_board[r][j] in same_list:
                    break;
                else:
                    if curr_board[r][j] == '.' or curr_board[r][j] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r, j, key)
                        #add to the list of possible board configurations    
                        temp.append(temp_board)
                        if curr_board[r][j] in opp_list:
                            break
                
            for j in range(c + 1, 8):
                if curr_board[r][j] in same_list:
                    break;        
                else:
                    if curr_board[r][j] == '.' or curr_board[r][j] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r, j, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[r][j] in opp_list:
                            break
        
        return temp                  
        pass
    
    #To find out the possible successors for all the bluejays of the current player.
    def move_bluejay(self, curr_board, positions):
        temp = []
        temp_board = []
        
        if self.current_player == 'w':
            key = 'B'
            same_list = pieces_names.keys()
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'b'
            same_list = map(lambda x: x.lower(), pieces_names.keys())
            opp_list = pieces_names.keys()
        
        for pos in positions:
            r, c = pos[0], pos[1]
            
            #upper left diagonal
            for i in range(r - 1, -1, -1):
                if c - (r - i) >= 0:
                    if curr_board[i][c - (r - i)] in same_list:
                        break;
                    else:
                        if curr_board[i][c - (r - i)] == '.' or curr_board[i][c - (r - i)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c - (r - i), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c - (r - i)] in opp_list:
                                break
                        
            #bottom right diagonal
            for i in range(r + 1, 8):
                if c + (i - r) <= 7:
                    if curr_board[i][c + (i - r)] in same_list:
                        break;
                    else:
                        if curr_board[i][c + (i - r)] == '.' or curr_board[i][c + (i - r)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c + (i - r), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c + (i - r)] in opp_list:
                                break
            
            #bottom left diagonal
            for i in range(r + 1, 8):
                if c - (i - r) >= 0:
                    if curr_board[i][c - (i - r)] in same_list:
                        break;
                    else:
                        if curr_board[i][c - (i - r)] == '.' or curr_board[i][c - (i - r)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c - (i - r), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c - (i - r)] in opp_list:
                                break
                            
            #top right diagonal
            for i in range(r - 1, -1, -1):
                if c + (r - i) <= 7:
                    if curr_board[i][c + (r - i)] in same_list:
                        break;
                    else:
                        if curr_board[i][c + (r - i)] == '.' or curr_board[i][c + (r - i)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c + (r - i), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c + (r - i)] in opp_list:
                                break
                            
        return temp    
        pass
    
    #To find out the possible successors for all the quetzals of the current player.
    def move_quetzal(self, curr_board, positions):
        temp = []
        temp_board = []
        
        if self.current_player == 'w':
            key = 'Q'
            same_list = pieces_names.keys()
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'q'
            same_list = map(lambda x: x.lower(), pieces_names.keys())
            opp_list = pieces_names.keys()
        
        for pos in positions:
            r, c = pos[0], pos[1]
            
            for i in range(r - 1, -1, -1):
                if curr_board[i][c] in same_list:
                    break;
                else:
                    if curr_board[i][c] == '.' or curr_board[i][c] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', i, c, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[i][c] in opp_list:
                            break
                        
            for i in range(r + 1, 8):
                if curr_board[i][c] in same_list:
                    break;
                else:
                    if curr_board[i][c] == '.' or curr_board[i][c] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', i, c, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[i][c] in opp_list:
                            break
                    
            for j in range(c - 1, -1, -1):
                if curr_board[r][j] in same_list:
                    break;
                else:
                    if curr_board[r][j] == '.' or curr_board[r][j] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r, j, key)
                        #add to the list of possible board configurations    
                        temp.append(temp_board)
                        if curr_board[r][j] in opp_list:
                            break
                
            for j in range(c + 1, 8):
                if curr_board[r][j] in same_list:
                    break;        
                else:
                    if curr_board[r][j] == '.' or curr_board[r][j] in opp_list:
                        temp_board = self.swap_blocks(curr_board, r, c, '.', r, j, key)
                        #add to the list of possible board configurations
                        temp.append(temp_board)
                        if curr_board[r][j] in opp_list:
                            break
        
            #upper left diagonal
            for i in range(r - 1, -1, -1):
                if c - (r - i) >= 0:
                    if curr_board[i][c - (r - i)] in same_list:
                        break;
                    else:
                        if curr_board[i][c - (r - i)] == '.' or curr_board[i][c - (r - i)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c - (r - i), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c - (r - i)] in opp_list:
                                break
                        
            #bottom right diagonal
            for i in range(r + 1, 8):
                if c + (i - r) <= 7:
                    if curr_board[i][c + (i - r)] in same_list:
                        break;
                    else:
                        if curr_board[i][c + (i - r)] == '.' or curr_board[i][c + (i - r)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c + (i - r), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c + (i - r)] in opp_list:
                                break
            
            #bottom left diagonal
            for i in range(r + 1, 8):
                if c - (i - r) >= 0:
                    if curr_board[i][c - (i - r)] in same_list:
                        break;
                    else:
                        if curr_board[i][c - (i - r)] == '.' or curr_board[i][c - (i - r)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c - (i - r), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c - (i - r)] in opp_list:
                                break
                            
            #top right diagonal
            for i in range(r - 1, -1, -1):
                if c + (r - i) <= 7:
                    if curr_board[i][c + (r - i)] in same_list:
                        break;
                    else:
                        if curr_board[i][c + (r - i)] == '.' or curr_board[i][c + (r - i)] in opp_list:
                            temp_board = self.swap_blocks(curr_board, r, c, '.', i, c + (r - i), key)
                            #add to the list of possible board configurations
                            temp.append(temp_board)
                            if curr_board[i][c + (r - i)] in opp_list:
                                break
                            
        return temp
        pass

    #To find out the possible successors for the kingfisher of the current player.
    def move_kingfisher(self, curr_board, positions):
        temp = []
        temp_board = []
        
        if self.current_player == 'w':
            key = 'K'
            opp_list = map(lambda x: x.lower(), pieces_names.keys())
        else:
            key = 'k'
            opp_list = pieces_names.keys()

        pos = positions[0]
        r, c = pos[0], pos[1]
        
        #move straight
        if r <= 6:
            if curr_board[r + 1][c] == '.' or curr_board[r + 1][c] in opp_list:
                temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c, key)
                temp.append(temp_board)
                
        if r >= 1:
            if curr_board[r - 1][c] == '.' or curr_board[r - 1][c] in opp_list:
                temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c, key)
                temp.append(temp_board)
                
        if c <= 6:
            if curr_board[r][c + 1] == '.' or curr_board[r][c + 1] in opp_list:
                temp_board = self.swap_blocks(curr_board, r, c, '.', r, c + 1, key)
                temp.append(temp_board)
                
        if c >= 1:
            if curr_board[r][c - 1] == '.' or curr_board[r][c - 1] in opp_list:
                temp_board = self.swap_blocks(curr_board, r, c, '.', r, c - 1, key)
                temp.append(temp_board)
                
        #move diagonally
        if r - 1 >= 0 and c + 1 < 8 and (curr_board[r - 1][c + 1] == "." or curr_board[r - 1][c + 1] in opp_list):
                temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c + 1, key)
                temp.append(temp_board)

        if r - 1 >= 0 and c - 1 >= 0 and (curr_board[r - 1][c - 1] == "." or curr_board[r - 1][c - 1] in opp_list):
            temp_board = self.swap_blocks(curr_board, r, c, '.', r - 1, c - 1, key)
            temp.append(temp_board)

        if r + 1 < 8 and c + 1 < 8 and (curr_board[r + 1][c + 1] == "." or curr_board[r + 1][c + 1] in opp_list):
            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c + 1, key)
            temp.append(temp_board)

        if r + 1 < 8 and c - 1 >= 0 and (curr_board[r + 1][c - 1] == "." or curr_board[r + 1][c - 1] in opp_list):
            temp_board = self.swap_blocks(curr_board, r, c, '.', r + 1, c - 1, key)
            temp.append(temp_board)
        
        return temp
        pass
    
    #To find out all the possible successors for the current player.
    def successors(self, curr_boardObj):
        
        curr_board = curr_boardObj.board
        
        p_pos = []
        n_pos = []
        r_pos = []
        b_pos = []
        q_pos = []
        k_pos = []
        temp = []
        
        for i in range(0, len(curr_board)):
            for j in range(0, len(curr_board)):
                if self.current_player == 'w':
                    if curr_board[i][j] == 'P':
                        p_pos.append((i, j))
                    if curr_board[i][j] == 'N':
                        n_pos.append((i, j))
                    if curr_board[i][j] == 'R':
                        r_pos.append((i, j))
                    if curr_board[i][j] == 'B':
                        b_pos.append((i, j))
                    if curr_board[i][j] == 'Q':
                        q_pos.append((i, j))
                    if curr_board[i][j] == 'K':
                        k_pos.append((i, j))
                
                elif self.current_player == 'b':
                    if curr_board[i][j] == 'p':
                        p_pos.append((i, j))
                    if curr_board[i][j] == 'n':
                        n_pos.append((i, j))
                    if curr_board[i][j] == 'r':
                        r_pos.append((i, j))
                    if curr_board[i][j] == 'b':
                        b_pos.append((i, j))
                    if curr_board[i][j] == 'q':
                        q_pos.append((i, j))
                    if curr_board[i][j] == 'k':
                        k_pos.append((i, j))
        
        if len(k_pos) == 0:
            return None
        else:
            temp.extend(self.move_kingfisher(curr_board, k_pos))
        if len(p_pos) > 0:
            temp.extend(self.move_parakeet(curr_board, p_pos))
        if len(n_pos) > 0:    
            temp.extend(self.move_nighthawk(curr_board, n_pos))
        if len(r_pos) > 0:    
            temp.extend(self.move_robin(curr_board, r_pos))
        if len(b_pos) > 0:            
            temp.extend(self.move_bluejay(curr_board, b_pos))
        if len(q_pos) > 0:            
            temp.extend(self.move_quetzal(curr_board, q_pos))
        
        return temp
        pass
        
    #To start the game. Main method to play the game.
    def play_game(self, curr_boardObj):
        for depth in range(1, self.max_depth):
            #best_val = self.minimax(curr_boardObj, depth)
            best_val = self.alpha_beta(curr_boardObj, depth)
            #print self.best_board.board
            self.printable_board(self.best_board.board)
            #print '\n'
        pass
    
    #To convert the 8*8 board into a string to display the final board/output.
    def printable_board(self, curr_board):
        curr_board.shape = (-1, )
        print ''.join(curr_board)
        pass
    
class Chess_Board:
    def __init__(self, board, value):
        self.board = board
        self.value = value
        pass
        
player_names = {"w": "white", 
                "b": "black"}
                
pieces_names = {"P": "parakeet",
                "R": "robin",
                "B": "bluejay",
                "N": "nighthawk",
                "Q": "quetzal",
                "K": "kingfisher"}
               
#current_player = 'w'
#start_state = ".NBQ.BNRpPPPPKPP.....P.............p............ppp.pppprnbqkbnr"
#start_state = '.RB..B.RPPP.KPPP..NP.N...q..P...pp.................ppppprnb.kbnr'

current_player = sys.argv[1]
start_state = sys.argv[2]
time_out = sys.argv[3]

if current_player not in player_names.keys():
    sys.exit(1)

if len(start_state) != 64:
    sys.exit(1)
    
initial_board = np.array([start_state[i] for i in range(len(start_state))])
initial_board.shape = (8, 8)

print initial_board
print 'Too much work!!! Thinking...'

initial_boardObj = Chess_Board(initial_board, 0)
chess_game = Chess_Graph(initial_boardObj, current_player, 16)
chess_game.play_game(initial_boardObj)
