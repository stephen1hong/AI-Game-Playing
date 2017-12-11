"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    opponent_player = game.get_opponent(player)
    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")
    
    penalty_weight =1.5 # penalty_weight for opponent on availabe moves
    play_move_count = len(game.get_legal_moves(player))
    opp_move_count = len(game.get_legal_moves(opponent_player))
    return float(play_move_count -penalty_weight * opp_move_count)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    opponent_player = game.get_opponent(player)
    #Center Location = x0,y0 
    y0,x0 = int((game.height - 1)/2), int((game.width -1)/2) 

    #get locations
    y1,x1 = game.get_player_location(player)
    y2,x2 = game.get_player_location(opponent_player)

    #player/opponent_player distance from center
    dist_1 = math.sqrt((x1-x0)**2 + (y1-y0)**2) #player distance
    dist_2 = math.sqrt((x2-x0)**2 + (y2-y0)**2) #opponent distance

    #distance score
    dist_score =float(dist_2 - dist_1)
    return dist_score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # We have moves to play. How many more than our opponent?
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")    

    opponent_player = game.get_opponent(player)
    player_moves_count = len(game.get_legal_moves(player))
    opp_moves_count = len(game.get_legal_moves(opponent_player))

    if player_moves_count==opp_moves_count:
        y0,x0 = int((game.height - 1)/2), int((game.width -1)/2) #center position
        y1,x1 = game.get_player_location(player)
        y2,x2 = game.get_player_location(opponent_player)
        player_distance =abs(y1 -y0) +abs(x1-x0)   #use Manhattan distance
        opp_distance =abs(y2-y0) + abs(x2-x0)
        return float(opp_distance -player_distance)/10.
    else:
        return float(player_moves_count - opp_moves_count)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()
            if depth ==0 or not legal_moves:
                score= self.score(game,self)#score =self.score(game,self.player)
                return score

            max_score = float("-inf")
            for move in legal_moves:#game.get_legal_moves():
                new_game = game.forecast_move(move)
                score = min_value(new_game, depth-1)
                if score > max_score:
                    max_score = score
                    best_move =move
            return max_score

        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()
            if depth ==0 or not legal_moves: 
                score= self.score(game,self)
                return score

            min_score = float("inf")
            for move in legal_moves:#game.get_legal_moves():
                new_game = game.forecast_move(move)
                score = max_value(new_game, depth-1)
                if score < min_score:
                    min_score = score
                    best_move =move
            return min_score

        best_move =(-1,-1)
        legal_moves = game.get_legal_moves()
        if depth <= 0 or not legal_moves:
            score= self.score(game,self)
            return score 
        best_value = float("-inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            temp =float(min_value(new_game, depth-1))
            if temp > best_value:
                best_value =temp
                best_move =move
        return best_move
        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        search_depth = 1
        best_move = (-1, -1)
        while True:
            try:
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1
            except SearchTimeout:
                return best_move
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def max_value(game, depth,alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()
            if depth <=0 or not legal_moves:
                score= self.score(game,self)
                return score

            max_score = float("-inf")
            for move in legal_moves:
                new_game = game.forecast_move(move)
                score = min_value(new_game, depth-1, alpha, beta)
                max_score = max(max_score,score)
                if max_score >=beta:
                    return max_score
                alpha = max(alpha, max_score)
            return max_score

        def min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            legal_moves = game.get_legal_moves()
            if depth <=0 or not legal_moves:
                score= self.score(game,self)
                return score
                           
            min_score = float("inf")
            for move in legal_moves:
                new_game = game.forecast_move(move)
                score = max_value(new_game, depth-1, alpha, beta)
                min_score = min(min_score,score)
                if min_score <=alpha:
                    return min_score
                beta = min(beta, min_score)
            return min_score

        best_move =(-1,-1)
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            score= self.score(game,self)
            return score

        best_value = float("-inf")
        for move in legal_moves:
            new_state = game.forecast_move(move)
            temp =float(min_value(new_state, depth -1, alpha, beta ))
            if temp > best_value:
                best_value =temp
                best_move =move
            alpha=max(alpha, best_value)
            if best_value >= beta:
                return best_move
        return best_move
                           
