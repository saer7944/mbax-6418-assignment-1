// Optional development check: npm install playwright; node verify_browser.cjs
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');const assert=require('assert/strict');const {pathToFileURL}=require('url');
(async()=>{
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});const page=await browser.newPage({viewport:{width:1440,height:1060},deviceScaleFactor:1});const externalRequests=[];page.on('request',r=>{if(/^https?:/.test(r.url()))externalRequests.push(r.url())});const errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.goto(pathToFileURL(path.join(__dirname,'dashboard.html')).href);
const report={checks:[],browser:'Chromium',viewport:{width:1440,height:1060}};
for(const mode of ['balanced','binary']){
 await page.locator('#'+mode).click();const run=JSON.parse(fs.readFileSync(path.join(__dirname,'results',mode+'_raw.json'),'utf8'));const m=run.metrics;assert.equal(await page.locator('#'+mode).getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#emotions').isVisible(),mode==='balanced');assert.equal(await page.locator('#emotion-filter').isVisible(),mode==='balanced');
 const starColors=await page.locator('#star-bars .bar').evaluateAll(es=>es.map(e=>getComputedStyle(e).backgroundColor));if(mode==='binary')assert.equal(starColors[2],starColors[0]);else assert.notEqual(starColors[2],starColors[0]);
 assert.equal(await page.locator('#accuracy').textContent(),(m.accuracy*100).toFixed(1)+'%');assert.equal(await page.locator('#balanced-accuracy').textContent(),(m.balanced_accuracy*100).toFixed(1)+'%');assert.equal(await page.locator('#macro-f1').textContent(),m.macro_f1.toFixed(3));
 const stars=[1,2,3,4,5].map(s=>run.reviews.filter(r=>r.rating===s).length);
 assert.deepEqual(await page.locator('#star-bars .bar').evaluateAll(es=>es.map(e=>Number(e.dataset.value))),stars);
 assert.deepEqual(await page.locator('#class-bars .bar').evaluateAll(es=>es.map(e=>Number(e.dataset.value))),m.classes.flatMap(c=>[m.per_class[c].support,m.per_class[c].predicted]));
 assert.deepEqual(await page.locator('#recall-bars .bar').evaluateAll(es=>es.map(e=>Number(e.dataset.value))),m.classes.map(c=>m.per_class[c].recall));
 const cells=await page.locator('#matrix td').evaluateAll(es=>es.map(e=>Number(e.dataset.count)));assert.deepEqual(cells,m.confusion_matrix.flat());
 for(const status of ['all','correct','mismatch'])for(const actual of ['all',...m.classes])for(const predicted of ['all',...m.classes]){
  await page.locator('#status').selectOption(status);await page.locator('#actual').selectOption(actual);await page.locator('#predicted').selectOption(predicted);
  const expected=run.reviews.filter(r=>(status==='all'||r.correct===(status==='correct'))&&(actual==='all'||r.actual===actual)&&(predicted==='all'||r.sentiment===predicted));
  assert.equal(await page.locator('#reviews tr').count(),expected.length);assert.equal(await page.locator('#live-count').textContent(),`${expected.length} / ${run.reviews.length} reviews`);
  assert.deepEqual(await page.locator('#reviews tr').evaluateAll(es=>es.map(e=>e.dataset.reviewId)),expected.map(r=>r.review_id));
 }
 report.checks.push(`${mode}: headline metrics, every matrix cell, all status/actual/prediction filter combinations and row IDs match saved output`);
}
await page.locator('#balanced').click();const b=JSON.parse(fs.readFileSync(path.join(__dirname,'results/balanced_results.json'),'utf8'));
assert.equal(await page.locator('#emotion-agreement').textContent(),(b.emotion_metrics.agreement_rate*100).toFixed(1)+'%');
const ems=['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','none'];assert.deepEqual(await page.locator('#emotion-bars .bar').evaluateAll(es=>es.map(e=>Number(e.dataset.value))),ems.flatMap(e=>[b.emotion_metrics.llm_counts[e]||0,b.emotion_metrics.nrc_counts[e]||0]));
for(const val of ['agree','disagree','none']){await page.locator('#emotion-filter').selectOption(val);assert.equal(await page.locator('#reviews tr').count(),b.reviews.filter(r=>val==='agree'?r.emotion_agrees:val==='disagree'?!r.emotion_agrees:r.nrc_emotion==='none').length)}
await page.locator('#balanced').click();await page.locator('#search').fill('GC-');assert.equal(await page.locator('#reviews tr').count(),150);await page.locator('#search').fill('NO_SUCH_REVIEW_XXXX');assert.equal(await page.locator('#reviews tr').count(),0);assert.equal(await page.locator('#empty').isVisible(),true);await page.locator('#search').fill('');
const collapsed=await page.locator('.bar').evaluateAll(es=>es.filter(e=>Number(e.dataset.value)>0&&e.getBoundingClientRect().width===0).length);assert.equal(collapsed,0);
await page.locator('#search').fill('  gc-006014  ');assert.equal(await page.locator('#reviews tr').count(),1);assert.equal(await page.locator('#reviews tr').getAttribute('data-review-id'),'GC-006014');
await page.locator('#reviews summary').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#reviews details').getAttribute('open'),'');await page.keyboard.press('Enter');assert.equal(await page.locator('#reviews details').getAttribute('open'),null);
await page.locator('#search').fill('   ');assert.equal(await page.locator('#reviews tr').count(),150);await page.locator('#search').fill('');
assert.equal(await page.evaluate(()=>readable('Nice<br />gift &amp; card')),'Nice\ngift & card');
assert.equal(await page.evaluate(()=>readable('<script>window.bad=true</script>Safe')),'Safe');assert.equal(await page.evaluate(()=>window.bad),undefined);
const firstText=await page.locator('#reviews .full-text').first().textContent();assert.ok(firstText.includes('\n'));assert.ok(!firstText.includes('<br'));
for(const status of ['correct','mismatch'])for(const emotion of ['agree','disagree','none']){await page.locator('#status').selectOption(status);await page.locator('#emotion-filter').selectOption(emotion);const expected=b.reviews.filter(r=>r.correct===(status==='correct')&&(emotion==='agree'?r.emotion_agrees:emotion==='disagree'?!r.emotion_agrees:r.nrc_emotion==='none'));assert.deepEqual(await page.locator('#reviews tr').evaluateAll(es=>es.map(e=>e.dataset.reviewId)),expected.map(r=>r.review_id));}
await page.locator('#balanced').click();
for(const width of [320,390,768,820,1024,1440]){await page.setViewportSize({width,height:1060});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`Page overflows at ${width}px`);}
await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(__dirname,'assets/dashboard-overview-current.png')});
await page.locator('#status').selectOption('mismatch');await page.locator('#reviews details').first().evaluate(e=>e.open=true);await page.locator('#live-count').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(__dirname,'assets/dashboard-evidence-current.png')});
await page.locator('#theme').click();assert.equal(await page.locator('body').getAttribute('data-theme'),'blue');
await page.setViewportSize({width:390,height:844});await page.locator('#balanced').click();await page.evaluate(()=>window.scrollTo(0,0));assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);await page.screenshot({path:path.join(__dirname,'assets/dashboard-mobile-current.png')});
assert.deepEqual(errors,[]);assert.deepEqual(externalRequests,[]);report.checks.push('Regression checks: trimmed ID search, readable HTML/entities, inert markup, keyboard review expansion, active-run accessibility state, binary star colors, combined emotion/status filters, no external requests, and layouts from 320 to 1440 pixels');report.checks.push('Emotion filters, search, empty state, nonzero chart widths, palette switch, mobile page width and no JavaScript errors');report.status='passed';report.dashboard_sha256=require('crypto').createHash('sha256').update(fs.readFileSync(path.join(__dirname,'dashboard.html'))).digest('hex');fs.writeFileSync(path.join(__dirname,'results/browser_checks.json'),JSON.stringify(report,null,2)+'\n');await browser.close();console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
