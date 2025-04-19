from enum import IntEnum

class PieceType(IntEnum):
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    WHITE_JOKER = 7
    WHITE_STAR = 8
    EMPTY = 0
    BLACK_PAWN = -1
    BLACK_KNIGHT = -2
    BLACK_BISHOP = -3
    BLACK_ROOK = -4
    BLACK_QUEEN = -5
    BLACK_KING = -6
    BLACK_JOKER = -7
    BLACK_STAR = -8


def get_legal_moves(board, from_position, to_position):
    """
    Get the legal moves for a piece on the board.
    :param board: The chess board.
    :param from_position: The position of the piece to move from.
    :param to_position: The position to move to.
    :return: A list of legal moves.
    """
    legal_moves = []
    piece = board[from_position[0]][from_position[1]]
    if piece == PieceType.EMPTY:
        return legal_moves

    # Check if the move is within the bounds of the board
    if 0 <= to_position[0] < len(board) and 0 <= to_position[1] < len(board[0]):
        target_piece = board[to_position[0]][to_position[1]]
        # Check if the target square is empty or occupied by an opponent's piece
        if target_piece == PieceType.EMPTY or target_piece != piece:
            legal_moves.append(to_position)

    return legal_moves