import random

from bot import board

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
        valid_moves = board.get_all_valid_moves(side)
        board_state = board.get_board_state()
        opponent = 'b' if side == 'w' else 'w'

        if not valid_moves:
            return None

        best_score = float('-inf')
        best_move = None

        # Evaluate material balance to decide aggression/defense
        my_material = self.compute_material(board_state, side)
        opp_material = self.compute_material(board_state, opponent)
        material_diff = my_material - opp_material
        losing = material_diff < -500  # If down by a major piece

        for move in valid_moves:
            cloned_board = board.clone()
            cloned_board.apply_move(move)
            cloned_state = cloned_board.get_board_state()

            score = self.evaluate_move(board_state, move, side, opponent)

            # Bonus for check
            if self.puts_king_in_check(cloned_board, opponent):
                score += 150

            # Avoid losing own king
            if self.leaves_king_in_check(cloned_board, side):
                score -= 9999
                continue  # never make suicidal move

            # Avoid hanging piece
            (_, _), (tr, tc) = move
            if self.would_be_captured(cloned_board, tr, tc, opponent):
                score -= self.PIECE_VALUES.get(board_state[tr][tc][1], 0) * 0.9

            # Defensive: prefer draws when losing
            if losing:
                score += self.defensive_bias(move, board_state, side, opponent)

            # Offensive: trade up when ahead
            elif material_diff > 400:
                score += self.trade_advantage(move, board_state)

            # Prefer centralization and mobility
            score += len(cloned_board.get_all_valid_moves(side)) * 1.5

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    def would_be_captured(self, board_state, move, opponent):
        """
        Check if the piece at move's destination can be captured next turn.
        """
        (_, _), (tr, tc) = move
        simulated_attackers = self.get_attackers(board_state, tr, tc, opponent)
        return bool(simulated_attackers)

    def avoids_capture(self, board_state, move, side):
        """
        Reward if current piece is under threat and move avoids it.
        """
        (fr, fc), (tr, tc) = move
        opponent = 'b' if side == 'w' else 'w'
        attackers = self.get_attackers(board_state, fr, fc, opponent)
        if attackers:
            new_attackers = self.get_attackers(board_state, tr, tc, opponent)
            return len(new_attackers) < len(attackers)
        return False

    def get_attackers(self, board, target_row, target_col, attacker_side):
        """
        Get a list of opponent moves that could capture target square.
        """
        threats = []
        opp_moves = board.get_all_valid_moves(attacker_side)
        for (fr, fc), (tr, tc) in opp_moves:
            if tr == target_row and tc == target_col:
                threats.append(((fr, fc), (tr, tc)))
        return threats

    
    def compute_material(self, board_state, side):
        total = 0
        for row in board_state:
            for piece in row:
                if piece and piece[0] == side:
                    total += self.PIECE_VALUES.get(piece[1], 0)
        return total

    def puts_king_in_check(self, board, opponent):
        king_pos = self.find_king(board.get_board_state(), opponent)
        if not king_pos:
            return False
        for move in board.get_all_valid_moves(self._opposite(opponent)):
            (_, _), (tr, tc) = move
            if (tr, tc) == king_pos:
                return True
        return False

    def leaves_king_in_check(self, board, side):
        king_pos = self.find_king(board.get_board_state(), side)
        if not king_pos:
            return True
        for move in board.get_all_valid_moves(self._opposite(side)):
            (_, _), (tr, tc) = move
            if (tr, tc) == king_pos:
                return True
        return False

    def would_be_captured(self, board, row, col, attacker):
        for move in board.get_all_valid_moves(attacker):
            (_, _), (tr, tc) = move
            if tr == row and tc == col:
                return True
        return False

    def trade_advantage(self, move, board_state):
        (fr, fc), (tr, tc) = move
        target = board_state[tr][tc]
        attacker = board_state[fr][fc]
        if target and attacker:
            return self.PIECE_VALUES[target[1]] - self.PIECE_VALUES[attacker[1]]
        return 0

    def defensive_bias(self, move, board_state, side, opponent):
        """ When losing, avoid complex exchanges, keep king safe, and try to simplify """
        score = 0
        (fr, fc), (tr, tc) = move
        piece = board_state[fr][fc]
        target = board_state[tr][tc]

        if piece and piece[1] == 'K':
            score += 20  # Keep king mobile
        if target:
            # Avoid major trades unless gaining
            if self.PIECE_VALUES.get(target[1], 0) > 500:
                score -= 30
        # Avoid moving into attack range
        if self.would_be_captured(board_state, tr, tc, opponent):
            score -= 40

        return score

    def _opposite(self, side):
        return 'b' if side == 'w' else 'w'


    
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