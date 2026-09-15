# Draft Annotation Guidelines for Customer Support Intents

> **Status**: `[FROZEN - PHASE 4]`
> Note: These guidelines serve as the baseline for human annotators during Phase 4 Golden Set creation. Guidelines and taxonomy boundaries are subject to evidence-based validation and refinement during initial golden set labeling.

## Core Annotation Rules
1. **Primary Intent Priority Rule**: Label according to the customer's **primary requested action or core problem**, NOT merely the nouns mentioned in the message.
   - *Example*: 'My order says delivered but I don't have it' $\rightarrow$ `package_missing_damaged` (Core problem: missing package, not tracking).
   - *Example*: 'I need to cancel my order before it arrives' $\rightarrow$ `cancellation_modification` (Core action: cancel order, not tracking).
2. **Evidence-Based Intent Assignment**: Choose a specific domain intent whenever sufficient evidence exists. Use `other_unclear` **ONLY when no defined intent can be assigned reliably**.
   - *Example*: 'Amazon, your driver left my package at the wrong house' $\rightarrow$ `package_missing_damaged` (Sufficient evidence).
   - *Example*: 'Amazon help' $\rightarrow$ `other_unclear` (Contextless fragment).
3. **Canonical Risk Priors**: Baseline prior risk ratings must be one of strictly canonical lowercase values: `low`, `medium`, or `high`.
4. **Real Data Constraint & Zero Leakage**: All tweet excerpts shown below are extracted directly from `data/processed/train.csv` with zero overlap in validation or test splits.

---

### Intent: `order_tracking_delivery` (Order Tracking & Delivery Status)
**Definition**: Inquiries regarding package shipping status, tracking numbers, estimated delivery dates, transit delays, or location updates.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Asking where order/package is located
- Requesting tracking number or shipping status updates
- Inquiring about late or delayed deliveries before arrival

**Exclusion Criteria**:
- Reports of missing/stolen packages marked as delivered (use package_missing_damaged)
- Address change requests (use cancellation_modification)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: . Item has not been delivered but tracking says it was handed to me over an hour ago... 2nd time this has happened. Sort it out https://t.co/42W82GcARk
- [Real Tweet Excerpt]: Where is my order? https://t.co/pXnKSCo2ex
- [Real Tweet Excerpt]: HEY, I have 30 packages that need to be refunded on nov 5th which is the estimated shipping time. fake tracking. never shipped.

---

### Intent: `package_missing_damaged` (Missing, Stolen, or Damaged Items)
**Definition**: Reports of packages marked delivered but not received, stolen items, broken goods, missing components, or incorrect items delivered.

**Baseline Risk Prior**: `medium`

**Inclusion Criteria**:
- Tracking says delivered but package is missing or left at wrong address
- Received damaged, broken, or defective item
- Missing items from multi-item order shipment
- Received wrong item or size

**Exclusion Criteria**:
- General transit delays before scheduled delivery time (use order_tracking_delivery)
- Refund requests for returned items (use refund_return_processing)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: Is it possible to prevent AMZL from delivering my packages moving forward? Stuff is either lost/stolen/broken EVERY time.
- [Real Tweet Excerpt]: What the hell ?! This boxed is beyond damaged. I wanted a new one! https://t.co/J36cokDm6l
- [Real Tweet Excerpt]: This is quality of the grinder that was delivered just now.Broken grinder stone and scratched product all over.Mere refnd engh??? https://t.co/FyCq2DSjnE

---

### Intent: `refund_return_processing` (Refund & Return Processing)
**Definition**: Requests to initiate returns, check refund status, request prepaid return labels, or inquire about refund drop-off locations.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Asking how to return an item
- Checking status of a pending refund
- Requesting return shipping label or barcode
- Inquiring about refund processing timeframe

**Exclusion Criteria**:
- Reporting damaged item upon arrival (use package_missing_damaged)
- Unauthorized credit card charges (use payment_billing_issues)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: UPS supposed to pick up a return package yesterday and bring return label... never showed.
- [Real Tweet Excerpt]: That's what m doing n u making me believe more that refund policy by Amazon in India is really poor..! Ur seller is even not taking calls nw
- [Real Tweet Excerpt]: Last mail rec'd on 09.10.2017 stating refund will be in my account by 14.10.2017 Bt same not rec'd.

---

### Intent: `cancellation_modification` (Order Cancellation & Change Requests)
**Definition**: Requests to cancel an active order before shipment, modify order items, change shipping addresses, or update delivery speed.

**Baseline Risk Prior**: `medium`

**Inclusion Criteria**:
- Requesting to cancel an order immediately
- Changing shipping address on a recent purchase
- Modifying item quantities or order specifications

**Exclusion Criteria**:
- Returning an item that has already been shipped or delivered (use refund_return_processing)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: how do I cancel my amazon prime account? I don’t use it anymore as no longer need it.
- [Real Tweet Excerpt]: If I cancel my prime membership early do I get some money refunded?
- [Real Tweet Excerpt]: I got an order cancellation notification for a cancellation I did not do. I even got a refund. Can u pls tell me what's going on?

---

### Intent: `prime_subscription_membership` (Prime Membership & Subscription Services)
**Definition**: Inquiries regarding Prime membership fees, benefits, automatic renewals, trial cancellations, student discounts, or Prime Video access.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Inquiring about Prime membership charges or auto-renewal
- Requesting Prime subscription cancellation
- Questions about Prime Video, Music, or Student benefits

**Exclusion Criteria**:
- Digital content playback errors or streaming app crashes (use digital_technical_support)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: hey I started a Amazon prime trial..i was charged $1..wats that about ?
- [Real Tweet Excerpt]: If I cancel my prime membership early do I get some money refunded?
- [Real Tweet Excerpt]: ive signed up to prime student about a week ago with 6 months free so why is it saying theres a problem with my card? https://t.co/EvBkqR1APB

---

### Intent: `payment_billing_issues` (Payment, Charges & Billing Disputes)
**Definition**: Inquiries about unrecognized charges, declined payment methods, promo codes, gift cards, invoice requests, or double billing.

**Baseline Risk Prior**: `medium`

**Inclusion Criteria**:
- Disputing double or unexpected charges on credit card
- Fixing declined payment methods or card authorization issues
- Applying gift cards, promotional codes, or balance issues

**Exclusion Criteria**:
- Refund inquiries for returned items (use refund_return_processing)
- Prime auto-renewal fee questions (use prime_subscription_membership)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: did the sonos play:1 promo end? I thought tomorrow was last day. Your site is not accepting the promo code. Please advise!
- [Real Tweet Excerpt]: I spoke to a nice lady about the double charge. She said it was an automatic repayment set on my account. This lady took it off but . . .
- [Real Tweet Excerpt]: I'm trying to use the Echo 3 pack for, but I am unable to use the promo code https://t.co/SVPZcVCv4F

---

### Intent: `digital_technical_support` (Digital Content & Technical Support)
**Definition**: Issues accessing Kindle ebooks, Prime Video streaming errors, Fire TV hardware/app crashes, Alexa/echo device setup, or digital code redemption.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Prime Video buffering or error codes
- Kindle book purchase not showing up on device
- Fire TV app crashing or audio sync problems
- Redeeming digital gift codes or software keys

**Exclusion Criteria**:
- Physical hardware damage shipped in box (use package_missing_damaged)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: Huge thank you to 😁 my new remote for my Firestick ❤ arrived...it worked and I'm back in business #HappyWeeyin
- [Real Tweet Excerpt]: firestick remote voice control works only for prime video but not inside other apps- like . Not efficient
- [Real Tweet Excerpt]: 実家に帰ったらAmazon Fire TVが生えててガルパン見放題になってた

---

### Intent: `account_security_access` (Account Access, Security & Verification)
**Definition**: Issues logging into accounts, forgotten passwords, 2-factor authentication (2FA) locks, suspicious activity, or account suspensions.

**Baseline Risk Prior**: `high`

**Inclusion Criteria**:
- Unable to log into Amazon account
- 2FA verification code not received or locked out
- Reporting unauthorized login attempts or compromised account
- Password reset link not working

**Exclusion Criteria**:
- Updating default shipping address (use cancellation_modification)
- Billing disputes (use payment_billing_issues)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: Yes, but I am not receiving access codes. I cannot login to the Hub website to check my data.
- [Real Tweet Excerpt]: Still waiting for the password reset maybe by #cybermonday https://t.co/4hX6ZAHHZi
- [Real Tweet Excerpt]: I can't sign on to my account. When I ask for a password reset no email comes with a code. Help!!

---

### Intent: `product_inquiry_availability` (Product Inquiries & Stock Availability)
**Definition**: Questions about product specifications, restock dates, compatibility, warranty details, or seller authorization.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Asking when an item will be back in stock
- Inquiring about item dimensions, features, or compatibility
- Manufacturer warranty terms and seller contact info

**Exclusion Criteria**:
- Technical troubleshooting for purchased digital/hardware devices (use digital_technical_support)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: Why don't u share tracking details. You have added Seller with half info. You don't have seller contact num
- [Real Tweet Excerpt]: So now you have back in stock but you don't bother to fulfil my subscribe and save? Utter c**p.
- [Real Tweet Excerpt]: Hi, will these Avril Lavigne's colored vinyls be back in stock on Amazon Germany soon? I wanna buy them https://t.co/kZWsbyMA0W

---

### Intent: `feedback_general_complaint` (General Feedback & Service Complaints)
**Definition**: General customer feedback regarding customer service experiences, driver behavior, website features, or policy complaints.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- Complaining about poor customer support experience
- Feedback on delivery driver conduct or packaging material
- General policy suggestions or feature complaints

**Exclusion Criteria**:
- Actionable requests for specific missing package resolution (use package_missing_damaged)

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: why do you continue to use I don’t get my items! How can “the door” sign for a package! Terrible service!
- [Real Tweet Excerpt]: ORDR# 406-8734146-2073128 not delivrd on time. contacted 4 persons yet no response or confirmation. terrible service.
- [Real Tweet Excerpt]: Glitches in app, rude representatives and faulty products.. Is still reliable? https://t.co/a85TUkBNMw

---

### Intent: `other_unclear` (Other / Ambiguous Inquiries)
**Definition**: Fragmented tweets, ambiguous requests lacking context, greetings without questions, or requests outside customer support scope.

**Baseline Risk Prior**: `low`

**Inclusion Criteria**:
- One-word or vague messages like 'Help', 'Hello', 'DM sent'
- Non-support social media mentions or chatter
- Inquiries lacking sufficient information to classify into primary taxonomy

**Exclusion Criteria**:
- Any query with sufficient actionable evidence matching primary domain intents

**Real Customer Tweet Excerpts (from train.csv)**:
- [Real Tweet Excerpt]: Hello Don't send a special edition game like this in just a cardboard sleeve. It clearly got crushed b/c bad packaging. https://t.co/zoEmj23Hh5
- [Real Tweet Excerpt]: Hi there, still no response from anyone!
- [Real Tweet Excerpt]: - Hi there, I contacted the store 4 business days ago but have not heard back. Is there a help email at amazon? I can't make a A-Z claim at this time.

---
