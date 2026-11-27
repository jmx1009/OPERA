# Specimen pickups from a scheduled laboratory van

A laboratory van makes eight timetable visits, numbered 0 through 7, at ten-minute intervals from 08:00 to 09:10. It finishes at the laboratory. Five clinics each have one sealed specimen for that morning's intake. The van carries one aircraft that can leave at a scheduled visit, collect one specimen and meet the van again at a later visit. Once aboard the van, the specimen travels to the laboratory. An outside courier can collect any clinic's specimen and deliver it on time for the fee below.

| Clinic | Courier fee (USD) |
| --- | ---: |
| Elm | 32 |
| Pine | 36 |
| Reed | 29 |
| Oak | 38 |
| Vale | 37 |

The following bookings meet the clinics' collection windows and the van's timetable. Each spans the entire interval from launch to recovery. The aircraft can launch again at its recovery visit. The energy shown covers the entire booking, and its flight fee includes all operating charges.

| Booking | Clinic | Launch visit | Recovery visit | Energy (Wh) | Flight fee (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | Elm | 0 | 2 | 200 | 7 |
| E2 | Elm | 3 | 4 | 150 | 8 |
| P1 | Pine | 0 | 1 | 200 | 9 |
| P2 | Pine | 2 | 4 | 200 | 6 |
| R1 | Reed | 1 | 3 | 250 | 6 |
| R2 | Reed | 3 | 5 | 150 | 9 |
| O1 | Oak | 4 | 5 | 200 | 8 |
| O2 | Oak | 5 | 7 | 250 | 7 |
| V1 | Vale | 4 | 6 | 200 | 6 |
| V2 | Vale | 6 | 7 | 200 | 7 |

The single battery starts full at 400 Wh. A booking requires its full energy at launch. On each ten-minute leg spent aboard the van, the aircraft can regain up to 150 Wh, subject to the 400 Wh capacity. Charging occurs only on these legs. Any charge remaining when the van reaches the laboratory is acceptable.

Choose exactly one collection method per clinic. Minimize the sum of flight fees and outside courier fees in USD. Report the bookings in launch order, the clinics assigned to the courier and the minimum total cost.
