t = input(f'Enter the text you wish to add backslash before % in:\n>')
out = ''
for i, letter in enumerate(t):
    if letter == '%':
        out = out + '\\%'
    else:
        out = out + letter

print(out)
