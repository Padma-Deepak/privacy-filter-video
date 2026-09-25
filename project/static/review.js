/* No external scripts, telemetry or remote assets. Frame pixels stay in browser memory. */
'use strict';
const $ = id => document.getElementById(id);
const previewReady = new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='/static/preview.js?v=1';script.onload=resolve;script.onerror=()=>reject(new Error('Could not load live preview renderer'));document.head.append(script);});
const csrf = document.querySelector('meta[name="csrf-token"]').content;
let job = null, analysis = null, choices = {}, manual = [], boxes = [], current = 0;
let picture = null, requestSequence = 0, playing = false, drawing = 'keep', drag = null, loadedFrame = -1;
let selected = {}, nativeVideo = false;
let revealOriginal=true, geometry=new Map(), geometryRequests=new Map(), geometryGeneration=0, videoCallback=null;
const previewToggle=document.createElement('button');previewToggle.id='toggle-preview';previewToggle.textContent='Show blur preview';previewToggle.type='button';$('play').after(previewToggle);
const previewNotice=document.createElement('p');previewNotice.className='fine';previewNotice.setAttribute('role','status');$('viewer').after(previewNotice);
function previewLabel(){previewToggle.textContent=revealOriginal?'Show blur preview':'View original / select another';previewNotice.textContent=revealOriginal?'Original view — draw around the people to keep visible.':'Live blur preview — other detected faces are masked. Export uses full-resolution filters.';}
previewToggle.addEventListener('click',()=>{video.pause();revealOriginal=!revealOriginal;previewLabel();paint();});
const video = $('video');
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const previewHelp=document.querySelector('.timeline + .fine');if(previewHelp)previewHelp.textContent='Select a face to preview the other detected faces blurred. Use View original to add someone else. Export applies full-resolution filters.';
function show(panel) { ['upload','progress','review','result'].forEach(p => $(p+'-panel').hidden = p !== panel); }
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function api(path, options={}) {
  const response = await fetch(path, {...options, headers: {'X-CSRF-Token':csrf, ...options.headers}});
  if (!response.ok) { let message = 'Request failed. The job may have expired; reload to start again.';
    try { message = (await response.json()).error || message; } catch (_) {}
    throw new Error(message);
  }
  return response;
}
const endpoint = suffix => `/api/jobs/${job}${suffix}`;
const jsonPost = data => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const pause = ms => new Promise(resolve=>setTimeout(resolve,ms));
$('file').addEventListener('change',()=>{ $('file-name').textContent = $('file').files[0]?.name || 'Choose a video or image'; });
$('upload-form').addEventListener('submit',async event=>{
  event.preventDefault(); error('');
  const data = new FormData(event.target), custom=$('custom-profile').value.trim();
  if(custom) { data.set('profile','custom'); data.set('custom',custom); }
  $('analyse-button').disabled=true;
  try { const result=await (await api('/api/jobs',{method:'POST',body:data})).json(); job=result.id;
    sessionStorage.setItem('privacy-job',job); show('progress'); await watch();
  } catch(e) { error(e.message); } finally { $('analyse-button').disabled=false; }
});
async function watch() {
  const watched=job;
  while(job===watched) {
    const state=await (await api(endpoint(''))).json();
    $('progress-title').textContent=state.state==='exporting'?'Applying your choices.':'Finding the details.';
    $('progress-text').textContent=`${state.state} · ${state.done} / ${state.total || '…'} frames`;
    $('progress').max=Math.max(1,state.total); $('progress').value=state.done;
    if(state.state==='ready') { await loadReview(); return; }
    if(state.state==='complete') { showResult(state); return; }
    if(state.state==='error') { error(state.error); show('upload'); return; }
    if(state.state==='cancelled') { reset(); return; }
    await pause(650);
  }
}
function reset() { geometryGeneration++;geometry.clear();geometryRequests.clear(); video.pause(); video.removeAttribute('src'); video.load(); selected={}; nativeVideo=false; playing=false; job=null; analysis=null; picture=null; choices={}; manual=[]; boxes=[]; loadedFrame=-1; sessionStorage.removeItem('privacy-job'); show('upload'); }
async function discard() {
  if(!job) return;
  try { await api(endpoint(''),{method:'DELETE'}); reset(); error(''); } catch(e) {error(e.message);}
}
['cancel','discard','new-job'].forEach(id=>$(id).addEventListener('click',discard));
async function loadReview() {
  await previewReady;
  analysis=await (await api(endpoint('/analysis'))).json();
  choices={}; manual=[]; selected={}; current=0; loadedFrame=-1; playing=false; drawing='keep'; nativeVideo=false; video.hidden=true;
  geometryGeneration++;geometry=new Map();geometryRequests=new Map();revealOriginal=true;previewLabel();
  $('reviewed').checked=false; $('manual-list').replaceChildren();
  $('seek').max=analysis.frame_count-1; $('seek').value=0; $('audio').value=analysis.profile.audio;
  $('track-count').textContent='(0)'; $('people').textContent='No one selected yet. Draw a box around a face in the video.';
  $('warnings').textContent=analysis.warnings.join(' '); $('warnings').hidden=!analysis.warnings.length;
  $('play').hidden=!analysis.is_video; $('tracks').replaceChildren();
  for(const track of Object.values(analysis.tracks)) addTrack(track);
  if(!Object.keys(analysis.tracks).length) $('tracks').textContent='No tracks found. Detection may have missed faces; use manual hide regions.';
  show('review'); await loadFrame(0); prefetchGeometry();
  canvas.style.cursor='crosshair'; $('selection-status').textContent='Draw around a face to keep that person visible.';
  if(analysis.is_video) { video.src=endpoint('/source'); video.load(); }

}
function addTrack(track) {
  const row=document.createElement('div'); row.className='track'; row.id='track-'+track.id;
  const img=document.createElement('img'); img.src=track.thumbnail; img.alt=track.id+' preview';
  const content=document.createElement('div'), title=document.createElement('strong'), meta=document.createElement('small');
  title.textContent=track.id; meta.textContent=`Frames ${track.first}–${track.last} · ${track.gap_frames} gap-filled`;
  const select=document.createElement('select'); select.setAttribute('aria-label',`Choice for ${track.id}`);
  for(const [value,label] of [['hide','Always hide'],['keep','Keep visible'],['range','Hide only in a frame range']]) { const option=new Option(label,value); select.add(option); }
  const range=document.createElement('div'); range.className='range'; range.hidden=true;
  const start=document.createElement('input'), end=document.createElement('input');
  for(const [input,label,value] of [[start,'Start',track.first],[end,'End',track.last]]) { input.type='number';input.min=0;input.max=analysis.frame_count-1;input.value=value;input.setAttribute('aria-label',`${label} frame for ${track.id}`);range.append(input); }
  function update() { choices[track.id]={mode:select.value};range.hidden=select.value!=='range';if(select.value==='range') Object.assign(choices[track.id],{start:Number(start.value),end:Number(end.value)});$('reviewed').checked=false;peopleList();paint(); }
  select.addEventListener('change',update);start.addEventListener('input',update);end.addEventListener('input',update);
  title.tabIndex=0;title.style.cursor='pointer';title.title='Jump to first frame';
  title.addEventListener('click',()=>loadFrame(track.first).catch(e=>error(e.message)));
  title.addEventListener('keydown',e=>{if(e.key==='Enter') title.click();});
  content.append(title,meta,select,range);row.append(img,content);$('tracks').append(row);
}
function setChoice(key,mode) { choices[key]={mode};const row=$('track-'+key);row.querySelector('select').value=mode;row.querySelector('.range').hidden=true;$('reviewed').checked=false; peopleList(); }
for(const [id,mode] of [['hide-all','hide'],['keep-all','keep'],['invert',null]]) $(id).addEventListener('click',()=>{ for(const key of Object.keys(analysis.tracks)) setChoice(key,mode || ((choices[key]?.mode || 'hide')==='hide'?'keep':'hide'));paint(); });
function hidden(key) {const choice=choices[key] || {mode:'hide'};return choice.mode==='hide' || choice.mode==='range' && current>=choice.start && current<=choice.end;}
async function loadFrame(index) {
  current=Math.max(0,Math.min(analysis.frame_count-1,index));const requested=current,seq=++requestSequence;
  $('seek').value=current; $('frame-label').textContent=`Frame ${current} / ${analysis.frame_count-1}`;
  if(nativeVideo) {
    video.pause(); video.currentTime=frameTime(requested); loadedFrame=requested;
    boxes=await frameBoxes(requested); if(seq!==requestSequence)return;paint();return;
  }
  const [imageResponse,boxResponse]=await Promise.all([api(endpoint(`/frames/${requested}`)),frameBoxes(requested)]);
  const [blob,newBoxes]=await Promise.all([imageResponse.blob(),Promise.resolve(boxResponse)]);
  const bitmap=await createImageBitmap(blob);
  if(seq!==requestSequence || !analysis){bitmap.close();return;}
  picture?.close();picture=bitmap;boxes=newBoxes;loadedFrame=requested;canvas.width=bitmap.width;canvas.height=bitmap.height;paint();
}
function rectangle(box,color,label,dashed=false) {
  const sx=canvas.width/analysis.width,sy=canvas.height/analysis.height;
  const [x,y,w,h]=box;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dashed?[6,4]:[]);ctx.strokeRect(x*sx,y*sy,w*sx,h*sy);ctx.setLineDash([]);
  ctx.font='12px sans-serif';const tw=ctx.measureText(label).width;ctx.fillStyle=color;ctx.fillRect(x*sx,Math.max(0,y*sy-19),tw+10,19);ctx.fillStyle='white';ctx.fillText(label,x*sx+5,Math.max(14,y*sy-5));
}
async function frameBoxes(index) {
  if(geometry.has(index))return geometry.get(index);
  if(geometryRequests.has(index))return geometryRequests.get(index);
  const generation=geometryGeneration,url=endpoint(`/frames/${index}?boxes=1`);
  const promise=api(url).then(response=>response.json()).then(records=>{if(generation===geometryGeneration)geometry.set(index,records);return records;}).finally(()=>{if(generation===geometryGeneration)geometryRequests.delete(index);});
  geometryRequests.set(index,promise);return promise;
}
async function prefetchGeometry() {
  const generation=geometryGeneration,count=analysis.frame_count;
  let next=0;
  async function worker(){while(next<count && generation===geometryGeneration && analysis){const index=next++;try{await frameBoxes(index);}catch(_){return;}}}
  await Promise.all(Array.from({length:4},worker));
}
function paint() {
  if(!analysis || !window.PrivacyPreview)return;
  const source=nativeVideo?video:picture;
  if(!source || (nativeVideo && video.readyState<2))return;
  const records=geometry.get(current);
  if(revealOriginal)ctx.drawImage(source,0,0,canvas.width,canvas.height);
  else if(records)PrivacyPreview.render(ctx,source,records,current,choices,manual,analysis.profile,analysis.width,analysis.height);
  else {
    // Never silently show unmasked video while this frame's masks are loading.
    ctx.fillStyle='#18342f';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='white';ctx.font='16px sans-serif';ctx.fillText('Loading blur preview…',20,35);
    frameBoxes(current).then(()=>paint()).catch(e=>{video.pause();error(e.message);});return;
  }
  for(const b of ($('show-boxes').checked && !playing ? (records||[]) : [])) rectangle(b.box,hidden(b.id)?'#c74332':'#16855b',b.id+(b.source==='gap_fill'?' · gap':''),b.source==='gap_fill');
  for(const [key,item] of Object.entries(selected)) {if((choices[key]?.mode)==='keep' && item.positions[current]) rectangle(item.positions[current],'#16855b',item.label);}
  manual.forEach((b,i)=>{if(current>=b.start&&current<=b.end)rectangle(b.box,'#b04cc2',`manual ${i+1}`);});
  if(drag?.end)rectangle([Math.min(drag.start[0],drag.end[0]),Math.min(drag.start[1],drag.end[1]),Math.abs(drag.end[0]-drag.start[0]),Math.abs(drag.end[1]-drag.start[1])],'#b04cc2','new region');
}
function paintVideoFrame(now,metadata){
  if(!analysis || !nativeVideo){videoCallback=null;return;}
  current=frameAt(metadata.mediaTime);loadedFrame=current;$('seek').value=current;
  paint();videoCallback=video.requestVideoFrameCallback(paintVideoFrame);
}
function point(event) {const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(analysis.width,(event.clientX-r.left)/r.width*analysis.width)),Math.max(0,Math.min(analysis.height,(event.clientY-r.top)/r.height*analysis.height))];}
canvas.addEventListener('pointerdown',event=>{
  if(!analysis || loadedFrame!==current || playing)return;
  video.pause();drag={start:point(event)};canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove',event=>{if(drag){drag.end=point(event);paint();}});
canvas.addEventListener('pointerup',async event=>{
  if(!drag)return;const end=point(event),start=drag.start;drag=null;
  const x=Math.floor(Math.min(start[0],end[0])),y=Math.floor(Math.min(start[1],end[1]));
  const w=Math.ceil(Math.max(start[0],end[0]))-x,h=Math.ceil(Math.max(start[1],end[1]))-y;
  if(w<=2||h<=2){paint();return;}
  if(drawing==='hide'){manual.push({box:[x,y,w,h],start:current,end:current});manualList();drawing='keep';$('selection-status').textContent='Hide region added. Adjust its frame range below.';}
  else {
    $('selection-status').textContent='Matching this face…';error('');
    try {
      const result=await (await api(endpoint('/select-face'),jsonPost({frame:current,box:[x,y,w,h]}))).json();
      const key=result.track.id;
      if(!selected[key])selected[key]={label:`Person ${Object.keys(selected).length+1}`,track:result.track,positions:Object.fromEntries(result.positions.map(p=>[p.frame,p.box]))};
      setChoice(key,'keep'); peopleList();revealOriginal=false;previewLabel();
      $('selection-status').textContent=`${selected[key].label} will stay visible along the matched track. Draw another box to add someone else.`;
    }catch(e){error(e.message);$('selection-status').textContent='No selection added. Try a tighter box or another frame.';}
  }
  $('reviewed').checked=false;paint();
});
canvas.addEventListener('pointercancel',()=>{drag=null;paint();});
$('draw').addEventListener('click',()=>{drawing='keep';video.pause();playing=false;revealOriginal=true;previewLabel();paint();canvas.style.cursor='crosshair';$('selection-status').textContent='Draw a tight box around the face you want to keep visible.';});
$('draw-hide').addEventListener('click',()=>{drawing='hide';video.pause();playing=false;canvas.style.cursor='crosshair';$('selection-status').textContent='Draw a fixed region to hide, then set its frame range.';});
$('show-boxes').addEventListener('change',()=>loadFrame(current).catch(e=>error(e.message)));
function peopleList(){
  const people=$('people');people.replaceChildren();let count=0;
  for(const [key,item] of Object.entries(selected)){
    if(choices[key]?.mode!=='keep')continue;count++;
    const row=document.createElement('div');row.className='track';
    const image=document.createElement('img');image.src=item.track.thumbnail;image.alt=item.label;
    const content=document.createElement('div'),name=document.createElement('strong'),detail=document.createElement('small'),remove=document.createElement('button');
    name.textContent=item.label;detail.textContent=`Visible for matched track · ${formatTime(frameTime(item.track.first))}–${formatTime(frameTime(item.track.last))}`;
    remove.textContent='Blur this person instead';remove.addEventListener('click',()=>{setChoice(key,'hide');peopleList();paint();});
    content.append(name,detail,remove);row.append(image,content);people.append(row);
  }
  if(!count)people.textContent='No one selected yet. Draw a box around a face in the video.';
  $('track-count').textContent=`(${count})`;
}
function formatTime(seconds){return `${Math.floor(seconds/60)}:${(seconds%60).toFixed(1).padStart(4,'0')}`;}
function frameTime(index){const ts=analysis.timestamps;return ts?.length===analysis.frame_count && ts[ts.length-1]>0 ? ts[index]-ts[0] : index/analysis.fps;}
function frameAt(time){let low=0,high=analysis.frame_count-1;while(low<high){const mid=Math.ceil((low+high)/2);if(frameTime(mid)<=time+0.002)low=mid;else high=mid-1;}return low;}
video.addEventListener('loadedmetadata',()=>{
  if(!analysis)return;nativeVideo=true;video.hidden=false;$('viewer').classList.add('native-video');
  const scale=Math.min(1,960/analysis.width);canvas.width=Math.round(analysis.width*scale);canvas.height=Math.round(analysis.height*scale);picture?.close();picture=null;loadedFrame=current;paint();
  if(video.requestVideoFrameCallback && videoCallback===null)videoCallback=video.requestVideoFrameCallback(paintVideoFrame);
});
video.addEventListener('loadeddata',()=>paint());
video.addEventListener('error',()=>{
  if(!analysis)return;nativeVideo=false;video.hidden=true;$('viewer').classList.remove('native-video');
  $('selection-status').textContent='This browser cannot play the original codec. You can still select faces in the frame preview.';
  loadFrame(current).catch(e=>error(e.message));
});
video.addEventListener('play',()=>{playing=true;$('play').textContent='Pause';});
video.addEventListener('pause',()=>{playing=false;$('play').textContent='Play';if(analysis&&nativeVideo){current=frameAt(video.currentTime);loadedFrame=current;paint();}});
video.addEventListener('seeked',()=>{if(analysis&&nativeVideo){current=frameAt(video.currentTime);loadedFrame=current;paint();}});
video.addEventListener('timeupdate',()=>{if(analysis&&nativeVideo){current=frameAt(video.currentTime);loadedFrame=current;$('seek').value=current;$('frame-label').textContent=`${formatTime(video.currentTime)} / ${formatTime(video.duration||0)}`;paint();}});
function manualList() {$('manual-list').replaceChildren();manual.forEach((region,index)=>{const row=document.createElement('div');row.className='manual-item';const label=document.createElement('span');label.textContent=`Region ${index+1} · frames`;row.append(label);
  ['start','end'].forEach(key=>{const input=document.createElement('input');input.type='number';input.min=0;input.max=analysis.frame_count-1;input.value=region[key];input.setAttribute('aria-label',`Manual region ${index+1} ${key} frame`);input.addEventListener('input',()=>{region[key]=Number(input.value);$('reviewed').checked=false;paint();});row.append(input);});
  const remove=document.createElement('button');remove.textContent='Remove';remove.addEventListener('click',()=>{manual.splice(index,1);manualList();paint();$('reviewed').checked=false;});row.append(remove);$('manual-list').append(row);});}
$('seek').addEventListener('input',()=>{playing=false;loadFrame(Number($('seek').value)).catch(e=>error(e.message));});
$('previous').addEventListener('click',()=>{playing=false;loadFrame(current-1).catch(e=>error(e.message));});
$('next').addEventListener('click',()=>{playing=false;loadFrame(current+1).catch(e=>error(e.message));});
$('play').addEventListener('click',async()=>{
  if(nativeVideo){if(video.paused){if(video.ended)video.currentTime=0;await video.play().catch(e=>error(e.message));}else video.pause();return;}
  playing=!playing;if(!playing)return;$('play').textContent='Pause';
  try{while(playing&&analysis&&current<analysis.frame_count-1){await loadFrame(current+1);await pause(1000/analysis.fps);}}catch(e){error(e.message);}finally{playing=false;$('play').textContent='Play';}
});
$('export').addEventListener('click',async()=>{
  error('');if(!$('reviewed').checked){error('Confirm that you reviewed the choices before exporting.');return;}
  video.pause();playing=false;
  try{await api(endpoint('/export'),jsonPost({choices,manual,audio:$('audio').value}));show('progress');await watch();}catch(e){error(e.message);}
});
function showResult(state) {geometryGeneration++;geometry.clear();video.pause();video.removeAttribute('src');video.load();nativeVideo=false;selected={};playing=false;picture?.close();picture=null;analysis=null;boxes=[];choices={};manual=[];$('tracks').replaceChildren();ctx.clearRect(0,0,canvas.width,canvas.height);show('result');$('download').href=endpoint('/download');$('report-json').href=endpoint('/report');$('report-html').href=endpoint('/report?format=html');$('report-json').hidden=$('report-html').hidden=!state.has_report;}
const previous=sessionStorage.getItem('privacy-job');if(previous){job=previous;show('progress');watch().catch(e=>{reset();error(e.message);});}
