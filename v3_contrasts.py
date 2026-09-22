"""Authored classification contrasts. Constructed labels, not Jev-generated answers."""
import random,string
import build_dataset as v
# Same sector, different event: business outcome versus a technical result.
NEWS_PAIRS=[
('The board of a chip manufacturer accepted an acquisition offer after shareholders approved the price. The deal will be funded with debt.', 'Engineers built a chip that uses a new interconnect to reduce communication delays. Tests compare its circuit design with existing processors.'),
('A cloud provider reported falling quarterly margins despite higher subscription revenue. Its shares dropped after the earnings call.', 'A cloud provider described a fault-tolerant storage protocol. Experiments show that it recovers corrupted blocks without losing acknowledged writes.'),
('A security software company agreed to buy a rival. The companies disclosed the purchase price and the expected closing date.', 'Researchers demonstrated a flaw in a security protocol and published a patch that prevents the attack.'),
('A smartphone maker raised its profit forecast after selling more handsets than expected. Analysts revised their earnings estimates.', 'A smartphone laboratory demonstrated a new battery chemistry. Repeated charge tests measured capacity retention and thermal stability.'),
('A robotics supplier filed for bankruptcy after creditors rejected its debt restructuring plan.', 'A robotics laboratory introduced a tactile sensor that distinguishes soft objects from rigid ones without a camera.'),
('An online retailer announced a stock buyback and a larger dividend. Investors welcomed the new capital allocation plan.', 'An online retailer published a new package-sorting algorithm. The study compares routing accuracy on a fixed warehouse dataset.'),
('A satellite operator secured a commercial loan to refinance bonds coming due next month. The agreement lowers its interest expense.', 'A satellite team demonstrated an optical communication link. The experiment measured the transmission rate and errors under atmospheric turbulence.'),
('An electric vehicle company priced a new share offering to raise cash for its factories. Existing shareholders will be diluted.', 'An electric vehicle research group tested a new inverter design that reduces switching losses at high voltage.'),
('A database vendor cut its annual revenue forecast after large customers delayed purchases. The announcement sent its stock lower.', 'A database team introduced an indexing method and compared query execution time against existing indexes.'),
('A gaming hardware company settled a commercial patent dispute by agreeing to pay a licensing fee.', 'A gaming hardware team presented a rendering technique that reduces aliasing while preserving frame rate.'),
('A telecommunications operator agreed to sell a subsidiary to reduce its outstanding debt.', 'A telecommunications research team demonstrated an error-correcting code that improves transmission reliability.'),
('A biotechnology equipment company reported a quarterly loss and announced a plan to cut operating expenses.', 'A laboratory developed an imaging instrument that resolves cell structures at a finer spatial scale than earlier equipment.')]

def tag(source,key,state,question,options,gold,group,prov):
 r=v.finalize(source,key,state,question,options,gold,'train',prov,group=group)
 return r

def build():
 rows=[]
 for i,pair in enumerate(NEWS_PAIRS):
  for key,text in zip(['2','3'],pair):
   rows.append(tag('news_contrast_v3',f'{i}:{key}',text,'Classify the main topic of this news article.',[(str(i),d) for i,d in enumerate(['World news and international affairs.','Sports.','Business and finance.','Science and technology.'])],key,f'news-contrast-v3:{i}',{'label_method':'authored economic event vs technical result','fictional':True,'pair':i}))
 # Distinguish entailment, contradiction, and extra unsupported detail from complete ledgers.
 # Six answerable/unknown claims share a state and stay in the same training group.
 rng=random.Random(31021)
 for i in range(100):
  entity=f'Workshop {1000+i}';n=rng.randint(4,80);t=rng.randint(1,15);other=n+rng.randint(1,5)
  state=f'{entity} shipped exactly {n} crates on day {t}. Every crate in that shipment was solid blue, with no other colors. The log gives no information about the crates\' contents or any other day.'
  claims=[(f'{entity} shipped at least {n} crates on day {t}.','entailed'),(f'{entity} shipped exactly {other} crates on day {t}.','contradicted'),(f'The crates shipped by {entity} on day {t} contained books.','neutral'),(f'No crate shipped by {entity} on day {t} was red.','entailed'),(f'At least one crate shipped by {entity} on day {t} was red.','contradicted'),(f'{entity} shipped {n} crates on day {t+1}.','neutral')]
  for j,(claim,gold) in enumerate(claims):
   rows.append(tag('entailment_contrast_v3',f'{i}:{j}',state,'Using only the passage, classify this claim: '+claim,[('entailed','Supported by the passage.'),('neutral','Neither supported nor contradicted; insufficient information.'),('contradicted','Contradicted by the passage.')],gold,f'entailment-contrast-v3:{i}',{'label_method':'complete count/color facts and explicitly unspecified content/other date','fictional':True,'count':n,'day':t,'claim_variant':j}))
 return rows
