import base64, re, sys
T='/home/claude/archify-kg/assets/template.html'
s=open('/home/claude/archify/archify/assets/template.html',encoding='utf-8').read()
F='/home/claude/fonttmp/node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-%s.woff2'
def face(w,style='normal'):
    b=base64.b64encode(open(F%f'{w}-{style}','rb').read()).decode()
    return "@font-face{font-family:'IBM Plex Mono';font-style:%s;font-weight:%s;font-display:swap;src:url(data:font/woff2;base64,%s) format('woff2');}"%(style,w,b)
css = "\n".join([face(400),face(500),face(600),face(400,'italic')]) + """
/* ---- kevingrainger.github.io theme: paper ground, ink blue accent, IBM Plex Mono ---- */
[data-theme="light"]{
  --bg:#f7f6f2; --grid:#eceae3; --canvas-dot:#e2e0d8;
  --text:#1b1b1f; --text-muted:#6a6a70; --text-dim:#8e8d92; --text-faint:#6a6a70;
  --panel:#fbfaf7; --panel-border:#dcdad3; --lane-fill:rgba(247,246,242,.55); --lane-stroke:#cfcdc5;
  --arrow:#8a8990; --arrow-emphasis:#2430bc; --mask:#fbfaf7;
  --backend-fill:rgba(36,48,188,.07);   --backend-stroke:#2430bc;
  --database-fill:rgba(27,27,31,.05);   --database-stroke:#1b1b1f;
  --external-fill:rgba(106,106,112,.09);--external-stroke:#6a6a70;
  --frontend-fill:rgba(46,107,79,.09);  --frontend-stroke:#2e6b4f;
  --security-fill:rgba(161,40,58,.08);  --security-stroke:#a1283a;
  --messagebus-fill:rgba(107,63,160,.08);--messagebus-stroke:#6b3fa0;
  --cloud-fill:rgba(61,122,140,.09);    --cloud-stroke:#3d7a8c;
  --toolbar-bg:rgba(251,250,247,.94); --toolbar-border:#dcdad3; --toolbar-text:#1b1b1f;
  --toolbar-hover:#ffffff; --toolbar-menu-bg:#fbfaf7;
}
body{font-family:'IBM Plex Mono',ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;}
h1{font-weight:500;letter-spacing:-.02em;}
.pulse-dot{display:none !important;}
.card h3{font-weight:500;}
[data-theme="light"] .diagram-container,[data-theme="light"] .card{box-shadow:none;border-radius:2px;}
"""
assert s.count('</head>')==1
s=s.replace('</head>','<style id="kg-theme">\n'+css+'</style>\n</head>')
old="theme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';"
assert s.count(old)==1; s=s.replace(old,"theme = 'light';")
n=s.count("font-family: 'JetBrains Mono', ui-monospace")
s=s.replace("font-family: 'JetBrains Mono', ui-monospace","font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace")
old2="return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';"
assert s.count(old2)==1; s=s.replace(old2,"return 'light';")
open(T,'w',encoding='utf-8').write(s); print('ok',n,len(s))
