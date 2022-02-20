
class Grid:
	def __init__(self):
		self.g = {
			'top-left': ' ',
			'top-center': ' ',
			'top-right': ' ',
			'middle-left': ' ',
			'middle-center': ' ',
			'middle-right': ' ',
			'bottom-left': ' ',
			'bottom-center': ' ',
			'bottom-right': ' '
		}
		
	def show_grid(self):
		print('\n-----')
		print(f"{self.g['top-left']}|{self.g['top-center']}|{self.g['top-right']}")
		print('-----')
		print(f"{self.g['middle-left']}|{self.g['middle-center']}|{self.g['middle-right']}")
		print('-----')
		print(f"{self.g['bottom-left']}|{self.g['bottom-center']}|{self.g['bottom-right']}")
		print('-----')
	
	def add_mark(self , ref, marker):
		if self.g[ref] != ' ':
			raise ValueError('This cell is already taken...')
		else:
			self.g[ref] = marker


def run_game():
	global flag
	global grid
	markers = 20 * ['X', 'O']
	
	print('Lets play tick tack toe!!!')
	grid.show_grid()
	
	while flag == True:
		player = markers.pop()
		player_turn(player, grid)
		winning_conditions(grid, player)


def player_turn(player, grid):
	while True:
			ref = input(f'\nPlayer ({player})\nEnter the grid ref you would like to select: ')
			
			if ref == 'q':
				print('\nQuitting programme...')
				global flag
				flag = False
				break

			try:
				grid.add_mark(ref, player)
				grid.show_grid()
				break
			except KeyError:
				print('\nIncorrectly formatted grid reference...')
			except ValueError:
				print('\nThe cell has has alreay been taken...')
			
			grid.show_grid()
	
	
def winning_conditions(grid, player):
	if grid.g['top-left'] == grid.g['top-center'] == grid.g['top-right'] and grid.g['top-left'] != ' ':
		winning_screen(player)
	elif grid.g['middle-left'] == grid.g['middle-center'] == grid.g['middle-right'] and grid.g['middle-left'] != ' ':
		winning_screen(player)
	elif grid.g['bottom-left'] == grid.g['bottom-center'] == grid.g['bottom-right'] and grid.g['bottom-left'] != ' ':
		winning_screen(player)
		
	elif grid.g['top-left'] == grid.g['middle-left'] == grid.g['bottom-left'] and grid.g['top-left'] != ' ':
		winning_screen(player)
	elif grid.g['top-center'] == grid.g['middle-center'] == grid.g['bottom-center'] and grid.g['top-center'] != ' ':
		winning_screen(player)
	elif grid.g['top-right'] == grid.g['middle-right'] == grid.g['bottom-right'] and grid.g['top-right'] != ' ':
		winning_screen(player)
		
	elif grid.g['top-left'] == grid.g['middle-center'] == grid.g['bottom-right'] and grid.g['top-left'] != ' ':
		winning_screen(player)
	elif grid.g['top-right'] == grid.g['middle-center'] == grid.g['bottom-left'] and grid.g['top-right'] != ' ':
		winning_screen(player)
		
	elif ' ' not in grid.g.values():
		stalemate_screen()
	else:
		pass


def winning_screen(player):
	message = f'Congratulations player {player}! You are the winner!!!'
	mess_len = len(message)
	print(mess_len * '-')
	print()
	print(message)
	print()
	print(mess_len * '-')
	play_again()


def stalemate_screen():
	message = 'Stalemate! There are no winners here...'
	mess_len = len(message)
	print(mess_len * '*')
	print()
	print(message)
	print()
	print(mess_len * '*')
	play_again()


def play_again():
	while True:
		user_play_again = input('\nWould you like to play again? (y/n) ')
		
		if user_play_again == 'n':
			global flag
			flag = False
			print('\nQuitting programme...')
			break
		elif user_play_again == 'y':
			global grid
			grid = Grid()
			break
		else:
			print('\nIncorrect value, please try again...')


flag = True
grid = Grid()
run_game()
