import random

class Bot:
    def __init__(self):
        # Piece values - adjusted for ACM Chess variant
        self.PIECE_VALUES = {
            'P': 100,    # Pawn
            'R': 500,    # Rook (Castle)
            'N': 320,    # Knight
            'B': 330,    # Bishop
            'Q': 900,    # Queen
            'K': 20000,  # King (extremely valuable since capturing wins)
            'S': 350,    # Star (slightly more valuable than Knight/Bishop)
            'J': 1000    # Joker (most powerful piece)
        }
        
        # Position tables for 6x6 board
        # Center control is highly valuable in this smaller board
        self.POSITIONAL_BONUS = {
            # Pawns: advance toward promotion
            'P': [
                [0, 0, 0, 0, 0, 0],       # promotion rank (becomes Joker)
                [100, 100, 100, 100, 100, 100],  # near promotion (high value)
                [50, 50, 50, 50, 50, 50],
                [10, 10, 20, 20, 10, 10],
                [5, 5, 10, 10, 5, 5],
                [0, 0, 0, 0, 0, 0]        # starting rank
            ],
            # Knight: center control
            'N': [
                [-50, -40, -30, -30, -40, -50],
                [-40, 0, 5, 5, 0, -40],
                [-30, 5, 10, 10, 5, -30],
                [-30, 5, 10, 10, 5, -30],
                [-40, 0, 5, 5, 0, -40],
                [-50, -40, -30, -30, -40, -50]
            ],
            # Bishop: access to diagonals
            'B': [
                [-20, -10, -10, -10, -10, -20],
                [-10, 5, 0, 0, 5, -10],
                [-10, 10, 10, 10, 10, -10],
                [-10, 10, 10, 10, 10, -10],
                [-10, 5, 0, 0, 5, -10],
                [-20, -10, -10, -10, -10, -20]
            ],
            # Rook: control files and ranks
            'R': [
                [0, 0, 0, 0, 0, 0],
                [5, 10, 10, 10, 10, 5],
                [-5, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, -5],
                [-5, 0, 0, 0, 0, -5],
                [0, 0, 5, 5, 0, 0]
            ],
            # Queen: center control with safety
            'Q': [
                [-20, -10, -10, -10, -10, -20],
                [-10, 0, 0, 0, 0, -10],
                [-10, 0, 5, 5, 0, -10],
                [-10, 0, 5, 5, 0, -10],
                [-10, 0, 0, 0, 0, -10],
                [-20, -10, -10, -10, -10, -20]
            ],
            # King: safety first - edges/corners preferred
            'K': [
                [30, 20, -10, -10, 20, 30],
                [20, 10, -20, -20, 10, 20],
                [-10, -20, -30, -30, -20, -10],
                [-10, -20, -30, -30, -20, -10],
                [20, 10, -20, -20, 10, 20],
                [30, 20, -10, -10, 20, 30]
            ],
            # Star: special piece - middle control and mobility
            'S': [
                [-40, -30, -20, -20, -30, -40],
                [-30, 0, 5, 5, 0, -30],
                [-20, 5, 15, 15, 5, -20],
                [-20, 5, 15, 15, 5, -20],
                [-30, 0, 5, 5, 0, -30],
                [-40, -30, -20, -20, -30, -40]
            ],
            # Joker: most powerful piece - center control
            'J': [
                [-10, -5, 0, 0, -5, -10],
                [-5, 0, 10, 10, 0, -5],
                [0, 10, 20, 20, 10, 0],
                [0, 10, 20, 20, 10, 0],
                [-5, 0, 10, 10, 0, -5],
                [-10, -5, 0, 0, -5, -10]
            ]
        }
        
    def move(self, side, board):
        """
        Determine the best move for the current side
        Args:
            side: 'w' for white, 'b' for black
            board: the chess board object
        Returns:
            ((from_row, from_col), (to_row, to_col)): the chosen move
        """
        # Get valid moves and board state
        valid_moves = board.get_all_valid_moves(side)
        board_state = board.get_board_state()
        opponent = 'b' if side == 'w' else 'w'
        
        # If no valid moves, return None (should not happen in this game)
        if not valid_moves:
            return None
            
        # First check for any moves that can capture the king and win immediately
        for move in valid_moves:
            (from_row, from_col), (to_row, to_col) = move
            target_piece = board_state[to_row][to_col]
            
            # If we can capture the king, do it immediately
            if target_piece and target_piece[0] == opponent and target_piece[1] == 'K':
                return move
                
        # Calculate scores for all moves
        move_scores = []
        
        for move in valid_moves:
            score = self.evaluate_move(board_state, move, side, opponent)
            move_scores.append((move, score))
            
        # Sort moves by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest scoring move
        return move_scores[0][0]
    
    def evaluate_move(self, board_state, move, side, opponent):
        """
        Evaluate the value of a move
        """
        (from_row, from_col), (to_row, to_col) = move
        score = 0
        
        # Get the moving piece and target piece
        moving_piece = board_state[from_row][from_col]
        target_piece = board_state[to_row][to_col]
        
        # Get the piece types
        moving_piece_type = moving_piece[1] if moving_piece else None
        
        # 1. Capture value (highest priority)
        if target_piece and target_piece[0] == opponent:
            target_piece_type = target_piece[1]
            score += 10 * self.PIECE_VALUES.get(target_piece_type, 0)
            
            # Extra points for capturing with less valuable pieces
            if moving_piece_type:
                score += (self.PIECE_VALUES.get(target_piece_type, 0) - 
                         self.PIECE_VALUES.get(moving_piece_type, 0) / 10)
        
        # 2. Positional improvement
        if moving_piece_type:
            # Current position value
            current_position_table = self.POSITIONAL_BONUS.get(moving_piece_type, 
                               [[0 for _ in range(6)] for _ in range(6)])
            
            # Adjust for side (white reads from bottom, black from top)
            if side == 'w':
                current_row = from_row
                new_row = to_row
            else:
                current_row = 5 - from_row
                new_row = 5 - to_row
                
            current_position_value = current_position_table[current_row][from_col]
            new_position_value = current_position_table[new_row][to_col]
            
            # Score the positional change
            position_improvement = new_position_value - current_position_value
            score += position_improvement
        
        # 3. Pawn promotion to Joker (very high priority)
        if moving_piece_type == 'P':
            if (side == 'w' and to_row == 0) or (side == 'b' and to_row == 5):
                score += self.PIECE_VALUES['J'] * 0.8  # Huge bonus for promotion
        
        # 4. Protect your king by keeping other pieces nearby
        # This is a simplified version - just add small bonus for pieces staying near the king
        king_pos = self.find_king(board_state, side)
        if king_pos:
            king_row, king_col = king_pos
            # Distance from king after move
            king_distance = abs(king_row - to_row) + abs(king_col - to_col)
            if king_distance <= 2:  # If staying close to king
                score += 10
        
        # 5. Control center squares
        if 1 <= to_row <= 4 and 1 <= to_col <= 4:
            score += 5
            if 2 <= to_row <= 3 and 2 <= to_col <= 3:  # Very center
                score += 5
        
        # 6. Avoid moving the king unless necessary
        if moving_piece_type == 'K':
            score -= 15  # Small penalty for moving king
            
            # Unless it's to capture something valuable
            if target_piece and target_piece[0] == opponent:
                target_piece_type = target_piece[1]
                if self.PIECE_VALUES.get(target_piece_type, 0) > 200:
                    score += 40  # Override penalty if capturing valuable piece
        
        return score
    
    def find_king(self, board_state, side):
        """Find the position of the king for the given side"""
        for row in range(6):
            for col in range(6):
                piece = board_state[row][col]
                if piece and piece[0] == side and piece[1] == 'K':
                    return (row, col)
        return None