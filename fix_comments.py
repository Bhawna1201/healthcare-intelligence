import re
src = open('dashboard.py').read()
src = re.sub(r'\s*<!--[^>]*-->\n', '\n', src)
open('dashboard.py', 'w').write(src)
print('Done — HTML comments removed')
