const {test}=require('node:test');
const assert=require('node:assert/strict');
const {regionsForFrame,render}=require('../../project/static/preview.js');
const profile={filters:{faces:'solid',plates:'solid'},padding:0.25};
const records=[{id:'faces:1',class:'faces',box:[10,10,20,20]},{id:'faces:2',class:'faces',box:[60,10,20,20]}];
test('selected face remains visible and every other detected face is masked',()=>{
 assert.deepEqual(regionsForFrame(records,0,{'faces:1':{mode:'keep'}},[],profile,100,64),[{box:[60,10,20,20],filter:'solid'}]);
});
test('multiple selected people are kept, new tracks still default to hidden',()=>{
 const three=[...records,{id:'faces:3',class:'faces',box:[0,40,10,10]}];
 assert.deepEqual(regionsForFrame(three,0,{'faces:1':{mode:'keep'},'faces:2':{mode:'keep'}},[],profile,100,64),[{box:[0,40,10,10],filter:'solid'}]);
});
test('frame ranges and manual hide regions match export precedence',()=>{
 const choices={'faces:1':{mode:'keep'},'faces:2':{mode:'range',start:2,end:4}};
 assert.equal(regionsForFrame(records,1,choices,[],profile,100,64).length,0);
 assert.equal(regionsForFrame(records,2,choices,[],profile,100,64).length,1);
 assert.deepEqual(regionsForFrame(records,1,choices,[{box:[10,10,20,20],start:1,end:1}],profile,100,64),[{box:[5,5,30,30],filter:'solid'}]);
});
test('preview actually paints hidden pixels instead of only drawing boxes',()=>{
 const calls=[];const context={canvas:{width:200,height:128},drawImage:(...args)=>calls.push(['source',...args]),fillRect:(...args)=>calls.push(['mask',...args])};
 const source={};render(context,source,records,0,{'faces:1':{mode:'keep'}},[],profile,100,64);
 assert.deepEqual(calls,[['source',source,0,0,200,128],['mask',120,20,40,40]]);
 assert.equal(context.fillStyle,'#000');
});
