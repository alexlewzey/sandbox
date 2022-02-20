"""demonstrating the mapping from character to ascii/unicode and from ascii/unicode to binary representation"""
chars = 'abcde'
encoding = [ord(char) for char in chars]
bits = [f'{idx:b}' for idx in encoding]
print(chars)
print(' '.join([str(idx) for idx in encoding]))
print(' '.join(bits))
