with open('agent.py', 'rb') as f:
    content = f.read()

fixed = content.decode('utf-8', errors='ignore')

with open('agent.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Fixed! Now run: python agent.py')