import numpy as np
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


def in_bounds(x, y):
    return 0 <= x < 6 and 0 <= y < 6


def is_opponent(p1, p2):
    return p1 * p2 < 0


def generate_sliding_moves(board, x, y, deltas):
    piece = board[x][y]
    moves = []
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        while in_bounds(nx, ny):
            if board[nx][ny] == PieceType.EMPTY:
                moves.append((nx, ny))
            elif is_opponent(piece, board[nx][ny]):
                moves.append((nx, ny))
                break
            else:
                break
            nx += dx
            ny += dy
    return moves


def generate_knight_moves(board, x, y):
    piece = board[x][y]
    deltas = [(2, 1), (1, 2), (-1, 2), (-2, 1),
              (-2, -1), (-1, -2), (1, -2), (2, -1)]
    return [(nx, ny) for dx, dy in deltas
            if in_bounds(nx := x + dx, ny := y + dy)
            and (board[nx][ny] == PieceType.EMPTY or is_opponent(piece, board[nx][ny]))]


def generate_king_moves(board, x, y):
    piece = board[x][y]
    moves = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and (board[nx][ny] == PieceType.EMPTY or is_opponent(piece, board[nx][ny])):
                moves.append((nx, ny))
    return moves


def generate_pawn_moves(board, x, y, is_white):
    direction = -1 if is_white else 1
    piece = board[x][y]
    moves = []

    # Forward 1
    nx, ny = x + direction, y
    if in_bounds(nx, ny) and board[nx][ny] == PieceType.EMPTY:
        moves.append((nx, ny))
        # Forward 2 if first square is free
        nx2 = x + 2 * direction
        if in_bounds(nx2, y) and board[nx2][y] == PieceType.EMPTY:
            moves.append((nx2, y))

    # Diagonal captures
    for dy in [-1, 1]:
        nx, ny = x + direction, y + dy
        if in_bounds(nx, ny) and is_opponent(piece, board[nx][ny]):
            moves.append((nx, ny))

    return moves


def generate_star_moves(board, x, y):
    piece = board[x][y]
    moves = []

    # 1-step diagonal
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny) and (board[nx][ny] == PieceType.EMPTY or is_opponent(piece, board[nx][ny])):
            moves.append((nx, ny))

    # 2-step jump (H/V, can jump)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny) and (board[nx][ny] == PieceType.EMPTY or is_opponent(piece, board[nx][ny])):
            moves.append((nx, ny))

    return moves


def generate_joker_moves(board, x, y):
    piece = board[x][y]
    moves = []
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and (board[nx][ny] == PieceType.EMPTY or is_opponent(piece, board[nx][ny])):
                moves.append((nx, ny))
    return moves


def get_legal_moves(board, from_pos):
    x, y = from_pos
    piece = board[x][y]
    if piece == PieceType.EMPTY:
        return []

    abs_piece = abs(piece)
    is_white = piece > 0

    if abs_piece == PieceType.WHITE_PAWN:
        return generate_pawn_moves(board, x, y, is_white)
    elif abs_piece == PieceType.WHITE_ROOK:
        return generate_sliding_moves(board, x, y, [(1, 0), (-1, 0), (0, 1), (0, -1)])
    elif abs_piece == PieceType.WHITE_BISHOP:
        return generate_sliding_moves(board, x, y, [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    elif abs_piece == PieceType.WHITE_QUEEN:
        return generate_sliding_moves(board, x, y, [(1, 0), (-1, 0), (0, 1), (0, -1),
                                                    (1, 1), (1, -1), (-1, 1), (-1, -1)])
    elif abs_piece == PieceType.WHITE_KNIGHT:
        return generate_knight_moves(board, x, y)
    elif abs_piece == PieceType.WHITE_KING:
        return generate_king_moves(board, x, y)
    elif abs_piece == PieceType.WHITE_STAR:
        return generate_star_moves(board, x, y)
    elif abs_piece == PieceType.WHITE_JOKER:
        return generate_joker_moves(board, x, y)

    return []
