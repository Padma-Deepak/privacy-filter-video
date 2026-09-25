/* Browser-only approximation of the export filters. No pixels leave the device. */
(function (root) {
  'use strict';
  function shouldHide(id, frame, choices) {
    const choice = choices[id] || {mode: 'hide'};
    return choice.mode === 'hide' || (choice.mode === 'range' && frame >= choice.start && frame <= choice.end);
  }
  function regionsForFrame(records, frame, choices, manual, profile, width, height) {
    const regions = records.filter(record => shouldHide(record.id, frame, choices))
      .map(record => ({box: record.box, filter: profile.filters[record.class]}));
    for (const region of manual) {
      if (frame < region.start || frame > region.end) continue;
      const [x,y,w,h] = region.box, px=w*profile.padding, py=h*profile.padding;
      const left=Math.max(0,x-px), top=Math.max(0,y-py);
      regions.push({box:[left,top,Math.min(width,x+w+px)-left,Math.min(height,y+h+py)-top],filter:profile.filters.faces});
    }
    return regions;
  }
  function render(context, source, records, frame, choices, manual, profile, width, height) {
    const cw=context.canvas.width,ch=context.canvas.height,sx=cw/width,sy=ch/height;
    context.drawImage(source,0,0,cw,ch);
    for(const region of regionsForFrame(records,frame,choices,manual,profile,width,height)) {
      const [x,y,w,h]=region.box;
      const left=Math.max(0,Math.floor(x*sx)),top=Math.max(0,Math.floor(y*sy));
      const rw=Math.min(cw-left,Math.ceil(w*sx)),rh=Math.min(ch-top,Math.ceil(h*sy));
      if(rw<=0||rh<=0)continue;
      if(region.filter==='solid') {context.fillStyle='#000';context.fillRect(left,top,rw,rh);continue;}
      // Crop into a separate surface so a filter never samples an adjacent face.
      const crop=document.createElement('canvas');crop.width=rw;crop.height=rh;
      const cc=crop.getContext('2d');cc.drawImage(context.canvas,left,top,rw,rh,0,0,rw,rh);
      if(region.filter==='blur' && typeof context.filter==='string') {
        const radius=Math.max(4,profile.blur_sigma*sx);
        context.save();context.beginPath();context.rect(left,top,rw,rh);context.clip();
        context.filter=`blur(${radius}px)`;
        // Extend the crop beyond its clip so transparent filter edges do not
        // simply expose the original underneath the preview mask.
        context.drawImage(crop,left-radius*2,top-radius*2,rw+radius*4,rh+radius*4);
        context.restore();
      } else {
        const small=document.createElement('canvas');
        const size=Math.max(8,profile.pixel_size*sx);
        small.width=Math.max(1,Math.round(rw/size));small.height=Math.max(1,Math.round(rh/size));
        small.getContext('2d').drawImage(crop,0,0,small.width,small.height);
        context.save();context.imageSmoothingEnabled=false;context.drawImage(small,left,top,rw,rh);context.restore();
      }
    }
  }
  root.PrivacyPreview={shouldHide,regionsForFrame,render};
  if(typeof module!=='undefined')module.exports=root.PrivacyPreview;
})(typeof window!=='undefined'?window:globalThis);
