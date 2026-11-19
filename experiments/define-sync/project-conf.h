#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define COOJA_MTARCH_STACKSIZE 16384

#define MAKE_MAC MAKE_MAC_TSCH
#define MAKE_ROUTING MAKE_ROUTING_RPL_LITE

/* RFC 8180 Section 6.1: MinHopRankIncrease = 256 */
#define RPL_CONF_MIN_HOPRANKINC 256

/* Standard RFC 6206 / RFC 6550 / RFC 6775 settings for dense single-cell 6TiSCH-MC
   (applied identically across DEFINE and all baselines) */
#define RPL_CONF_DIO_INTERVAL_MIN 13                /* 2^13 ms = 8.192 s */
#define RPL_CONF_DIO_INTERVAL_DOUBLINGS 8
#define RPL_CONF_DIO_REDUNDANCY 3                   /* RFC 6206 Trickle suppression (k=3) */
#define RPL_CONF_DIS_INTERVAL (120 * CLOCK_SECOND)  /* Prevent simultaneous DIS storms on (0,0) */
#define RPL_CONF_WITH_DAO_ACK 0                     /* Upward DODAG join matching Fig. 3 model */
#define RPL_CONF_PROBING_INTERVAL (180 * CLOCK_SECOND)
#define UIP_CONF_ND6_SEND_NS 0
#define UIP_CONF_ND6_SEND_NA 0
#define TSCH_CONF_MAC_MIN_BE 2
#define TSCH_CONF_MAC_MAX_BE 7

#ifndef SYNC_SCHEME
#define SYNC_SCHEME 1
#endif

#ifndef SF_LENGTH
#define SF_LENGTH 61
#endif

#ifndef NUM_CHANNELS
#define NUM_CHANNELS 16
#endif

#ifndef EB_MULT
#define EB_MULT 1
#endif

#ifndef ATP_SUBSLOTS
#define ATP_SUBSLOTS 2
#endif

#ifndef NBR_CAP
#define NBR_CAP 32
#endif

#ifndef POISSON_ARRIVAL
#define POISSON_ARRIVAL 0
#endif

#define TSCH_SCHEDULE_CONF_DEFAULT_LENGTH SF_LENGTH
#define TSCH_CONF_DEFAULT_TIMESLOT_LENGTH 10000 /* 10 ms */
#define ENERGEST_CONF_ON 1

#define TSCH_CALLBACK_NEW_TIME_SOURCE define_on_new_time_source

#define ONE_SF_PERIOD (SF_LENGTH * (CLOCK_SECOND / 100))
#define BASE_SF_PERIOD (EB_MULT * ONE_SF_PERIOD)

/* Identical neutral boot defaults for all schemes; protocol-specific rates are applied in node.c */
#define TSCH_CONF_EB_PERIOD (2 * BASE_SF_PERIOD)
#define TSCH_CONF_MAX_EB_PERIOD (32 * BASE_SF_PERIOD)
#define TSCH_CONF_CHANNEL_SCAN_DURATION ONE_SF_PERIOD

#define TSCH_PACKET_CONF_EB_WITH_SLOTFRAME_AND_LINK 1

/* Support 8-point channel granularity: 2, 4, 6, 8, 10, 12, 14, 16 */
#if NUM_CHANNELS == 2
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25 }
#elif NUM_CHANNELS == 4
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE TSCH_HOPPING_SEQUENCE_4_4
#elif NUM_CHANNELS == 6
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17 }
#elif NUM_CHANNELS == 8
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19 }
#elif NUM_CHANNELS == 10
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19, 11, 14 }
#elif NUM_CHANNELS == 12
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19, 11, 14, 18, 23 }
#elif NUM_CHANNELS == 14
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE (uint8_t[]){ 15, 25, 26, 20, 13, 17, 22, 19, 11, 14, 18, 23, 12, 16 }
#else
#define TSCH_CONF_DEFAULT_HOPPING_SEQUENCE TSCH_HOPPING_SEQUENCE_16_16
#endif

#define NBR_TABLE_CONF_MAX_NEIGHBORS NBR_CAP
#define NETSTACK_MAX_ROUTE_ENTRIES 512
#define TSCH_QUEUE_CONF_MAX_NEIGHBOR_QUEUES 64
#define TSCH_SCHEDULE_CONF_MAX_LINKS 64
#define TSCH_CONF_MAX_INCOMING_PACKETS 32
#define QUEUEBUF_CONF_NUM 64
#define UIP_CONF_BUFFER_SIZE 240
#define UIP_CONF_UDP 1
#define UIP_CONF_TCP 0
#define SICSLOWPAN_CONF_FRAG 0
#define TSCH_LOG_CONF_PER_SLOT 0

#include "node.h"

#endif /* PROJECT_CONF_H_ */