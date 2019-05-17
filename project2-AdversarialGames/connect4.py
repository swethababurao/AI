# connect4.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

"""
In this assignment, the task is to implement adversarial search algorithms with depth
limit for a Connect-4 game.

To complete the assignment, you must finish these functions:
    minimax (line 161), alphabeta (line 196), and expectiminimax (line 235)
in this file.

In the Connect-4 game, two players place discs in a 6-by-7 board one by one.
The discs will fall straight down and occupy the lowest available space of
the chosen column. The player wins if four of his or her discs are connected
in a line horizontally, vertically or diagonally.
See https://en.wikipedia.org/wiki/Connect_Four for more about Connect-4 games.


A Board() class is provided to simulate the game board.
It has the following properties:
    b.rows          # number of rows of the game board
    b.cols          # number of columns of the game board
    b.PLAYER1       # an integer flag to represent the player 1
    b.PLAYER2       # an integer flag to represent the player 2
    b.EMPTY_SLOT    # an integer flag to represent an empty slot in the board;

and the following methods:
    b.terminal()            # check if the game is terminal
                            # terminal means draw or someone wins
    b.has_draw()            # check if the game is a draw
    w = b.who_wins()        # return the winner of the game or None if there
    assert(w in [b.PLAYER1, b.PLAYER2, None])   # no winner yet?

    b.occupied(row, col)    # check if the slot at the specific location is
                            # occupied
    x = b.get(row, col)     # get the player occupying the given slot
    assert(x in [b.PLAYER1, b.PLAYER2, b.EMPTY_SLOT])
    row = b.row(r)          # get the specific row of the game described using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT
    col = b.column(r)       # get a specific column of the game board

    b.placeable(col)        # check if a checker can be placed at the specific
                            # column
    b.place(player, col)    # place a checker at the specific column for player
        # raise ValueError if the specific column does not have available space
    
    new_board = b.clone()   # return a new board instance having the same
                            # checker placement with b

    str = b.dump()          # a string to describe the game board using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT
"""

# use math library if needed
import math

def get_child_boards(player, board):
    """
    Generate a list of succesor boards obtained by placing a checker for a given player

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player who will place a checker in the board
    board: the current board instance

    Returns
    -------
    a list of (col, new_board) tuples,
    where col is the column in which a new checker is placed and
    is counted from the left column as 0, and
    new_board is the resulting board instance
    """
    res = []
    for c in range(board.cols):
        if board.placeable(c):
            tmp_board = board.clone()
            tmp_board.place(player, c)
            res.append((c, tmp_board))
    return res


def evaluate(player, board):
    """
    This is a function to evaluate the advantage of the specific player at the
    given game board.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the specific player
    board: the board instance

    Returns
    -------
    score: float
        a scalar to evaluate the advantage of the specific player at the given
        game board
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
### You can change the below and use your own evaluate function ###############
###############################################################################
    # Initialize the value of scores
    # [s0, s1, s2, s3, --s4--]
    # s0 for the case where all slots are empty in a 4-slot segment
    # s1 for the case where the player occupies one slot in a 4-slot line, the rest are empty
    # s2 for two slots occupied
    # s3 for three
    # s4 for four
    score = [0]*5
    adv_score = [0]*5

    # Initialize the weights
    # [w0, w1, w2, w3, --w4--]
    # w0 for s0, w1 for s1, w2 for s2, w3 for s3
    # w4 for s4
    weights = [0, 1, 4, 16, 1000]

    # Obtain all 4-slot segments on the board
    seg = []
    invalid_slot = -1
    left_revolved = [
        [invalid_slot]*r + board.row(r) + \
        [invalid_slot]*(board.rows-1-r) for r in range(board.rows)
    ]
    right_revolved = [
        [invalid_slot]*(board.rows-1-r) + board.row(r) + \
        [invalid_slot]*r for r in range(board.rows)
    ]
    for r in range(board.rows):
        # row
        row = board.row(r) 
        for c in range(board.cols-3):
            seg.append(row[c:c+4])
    for c in range(board.cols):
        # col
        col = board.col(c) 
        for r in range(board.rows-3):
            seg.append(col[r:r+4])
    for c in zip(*left_revolved):
        # slash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    for c in zip(*right_revolved): 
        # backslash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    # compute score
    for s in seg:
        if invalid_slot in s:
            continue
        if adversary not in s:
            score[s.count(player)] += 1
        if player not in s:
            adv_score[s.count(adversary)] += 1
    reward = sum([s*w for s, w in zip(score, weights)])
    penalty = sum([s*w for s, w in zip(adv_score, weights)])
    value = reward - penalty
###############################################################################
    return value

def maxValue(board,depth_limit):
    player = board.PLAYER1
    v = -math.inf 
    c = -1
    if depth_limit == 0:
        return evaluate(player, board), c
    print('MAX depth limit is not zero') 
    sv = {}   
    count = 0
    for a in get_child_boards(player, board):
        c, tmp_board = a
        print('Initial Value of V is', v)
        vv, cc = minValue(tmp_board,depth_limit-1)
        print('vv is', vv)
        v = max(v, vv)
        print('V is max(v, MIN)', v)
        sv[a] = v
        count+=1
    print('for loop count', count)
    if sv:
        key = max(sv, key=sv.get)
        c, tmp_board = key
        v = sv[key]
    print('finally evaluated value v in MAX', v)
    print('finally evaluated value c in MAX', c)
    return v, c

def minValue(board,depth_limit):
    player = board.PLAYER2  
    v = math.inf
    c = -1
    if depth_limit == 0:
        return evaluate(player, board),c
    print('MIN depth limit is not zero') 
    sv = {}    
    for a in get_child_boards(player, board):
        c, tmp_board = a
        #v = evaluate(player, tmp_board)
        print('Initial Value of V is', v)
        vv, cc = maxValue(tmp_board,depth_limit-1)
        print('vv is', vv)
        v = min(v, vv)
        print('V is min(v, MAX)', v)
        sv[a] = v
    if sv:
        key = min(sv, key=sv.get)
        c, tmp_board = key
        v = sv[key]
    print('finally evaluated value v in MIN', v)
    print('finally evaluated value c in MIN', c)
    return v, c


def minimax(player, board, depth_limit, maxing_player=True):
    """
    Minimax algorithm with limited depth

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to make an action (place a checker in the game)
    board: the current game board instance
    depth_limit: int
        the depth that the search algorithm needs to go further before stopping
    maxing_player: boolean

    Returns
    -------
    placement: int or None
        the column in which a checker should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    score: float
        a scalar to give an evaluation of the expected result led by the
        placement
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    placement = None

### Please finish the code below ##############################################
###############################################################################
    # Please initialize score value properly here or in the following code
    score = 0; 
    
    if not board.terminal():
        if depth_limit == 0:
            score = evaluate(adversary, board)
        elif maxing_player is True:
            print(maxing_player)
            v = -math.inf 
            c = -1
            sv = {}   
            for a in get_child_boards(player, board):
                c, tmp_board = a
                print('Initial Value of V is', v)
                cc, vv = minimax(adversary,tmp_board,depth_limit-1,maxing_player=False)
                print('vv is', vv)
                v = max(v, vv)
                print('V is max(v, MIN)', v)
                sv[a] = v
            if sv:
                key = max(sv, key=sv.get)
                c, tmp_board = key
                v = sv[key]
            print('finally evaluated value v in MAX', v)
            print('finally evaluated value c in MAX', c)
            score = v
            placement = c
        elif maxing_player is False:
            print(maxing_player)
            v = math.inf
            c = -1
            sv = {}    
            for a in get_child_boards(player, board):
                c, tmp_board = a
                print('Initial Value of V is', v)
                cc, vv = minimax(adversary,tmp_board,depth_limit-1,maxing_player=True)
                print('vv is', vv)
                v = min(v, vv)
                print('V is min(v, MAX)', v)
                sv[a] = v
            if sv:
                key = min(sv, key=sv.get)
                c, tmp_board = key
                v = sv[key]
            print('finally evaluated value v in MIN', v)
            print('finally evaluated value c in MIN', c)
            score = v
            placement = c               
   # if not board.terminal():
   #         if player == board.PLAYER1:
   #             #max implementation
   #            score, placement = maxValue(board,depth_limit)
   #         elif player == board.PLAYER2:
   #             #min implementation
   #            score, placement = minValue(board,depth_limit)
        
###############################################################################
    return placement, score    


def alphabeta(
        player, board, depth_limit,
        alpha=-math.inf, beta=math.inf, maxing_player=True):
    """
    Minimax algorithm with alpha-beta pruning.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to make an action (place a checker in the game)
    board: the current game board instance
    depth_limit: int
        the depth that the search algorithm needs to go further before stopping
    alpha: float
    beta: float
    maxing_player: boolean

    Returns
    -------
    placement: int or None
        the column in which a checker should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    score: float
        a scalar to give an evaluation of the expected result based on the
        placement
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    placement = None
    
### Please finish the code below ##############################################
###############################################################################
    # Please initialize score value properly here or in the following code
    score = 0
    if not board.terminal():
        if depth_limit == 0:
            score = evaluate(adversary, board)
        elif maxing_player is True:
            print(maxing_player)
            v = -math.inf 
            c = -1
            sv = {}   
            for a in get_child_boards(player, board):
                c, tmp_board = a
                cc, vv = alphabeta(adversary,tmp_board,depth_limit-1,alpha, beta,maxing_player=False)
                v = max(v, vv)
                sv[a] = v
                alpha = max(alpha, v)
                print('alpha is ',alpha)
                #if beta <= alpha:
                 #   return c,v
                if v>=beta:
                    print('inside v>=beta')
                    return c,v  
            if sv:
                key = max(sv, key=sv.get)
                c, tmp_board = key
                v = sv[key]
            score = v
            placement = c
        elif maxing_player is False:
            print(maxing_player)
            v = math.inf
            c = -1
            sv = {}    
            for a in get_child_boards(player, board):
                c, tmp_board = a
                cc, vv = alphabeta(adversary,tmp_board,depth_limit-1,alpha, beta,maxing_player=True)
                v = min(v, vv)
                sv[a] = v
                beta = min(beta, v)
                print('beta value is ', beta)
                #if beta <= alpha:
                #    return c,v
                if v<=alpha:
                    print('inside v<=alpha')
                    return c,v 
            if sv:
                key = min(sv, key=sv.get)
                c, tmp_board = key
                v = sv[key]
            score = v
            placement = c 
               
###############################################################################
    return placement, score


def expectimax(player, board, depth_limit, maxing_player=True):
    """
    Expectimax algorithm.
    We assume that the adversary of the initial player chooses actions
    uniformly at random.
    Say that it is the turn for Player 1 when the function is called initially,
    then during search, Player 2 is assumed to pick actions uniformly at
    random.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to make an action (place a checker in the game)
    board: the current game board instance
    depth_limit: int
        the depth that the search algorithm needs to go before stopping
    maxing_player: boolean

    Returns
    -------
    placement: int or None
        the column in which a checker should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    score: float
        a scalar to give an evaluation of the expected result based on the
        placement
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    placement = None

### Please finish the code below ##############################################
###############################################################################
    # Please initialize score value properly here or in the following code
    score = 0
    
    if not board.terminal():
        if depth_limit == 0:
            score = evaluate(adversary, board)
        elif maxing_player is True:
            print(maxing_player)
            v = -math.inf 
            c = -1
            sv = {}   
            for a in get_child_boards(player, board):
                c, tmp_board = a
                cc, vv = expectimax(adversary,tmp_board,depth_limit-1,maxing_player=False)
                v = max(v, vv)
                sv[a] = v
            if sv:
                key = max(sv, key=sv.get)
                c, tmp_board = key
                v = sv[key]
            score = v
            placement = c
        elif maxing_player is False:
            print(maxing_player)
            v = 0
            c = -1   
            for a in get_child_boards(player, board):
                c, tmp_board = a
                cc, vv = expectimax(adversary,tmp_board,depth_limit-1,maxing_player=True)   
                noofcols = 
                print('number of colums is:', noofcols)
                v = v + (1/7) * vv
            score = v
            placement = c   

###############################################################################
    return placement, score


if __name__ == "__main__":
    from utils.app import App
    import tkinter

    algs = {
        "Minimax": minimax,
        "Alpha-beta pruning": alphabeta,
        "Expectimax": expectimax
    }

    root = tkinter.Tk()
    App(algs, root)
    root.mainloop()
