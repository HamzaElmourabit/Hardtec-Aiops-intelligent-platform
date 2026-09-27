WITH tickets AS (

    SELECT
        TICKET_ID,
        TICKET_TEXT,
        TICKET_TYPE,
        PRIORITY,
        QUEUE,
        LANGUAGE
    FROM {{ ref('silver_tickets') }}

),

analytics AS (

    SELECT
        TICKET_TYPE,
        PRIORITY,
        QUEUE,

        COUNT(*) AS TOTAL_TICKETS,

        COUNT_IF(PRIORITY = 'high') AS HIGH_PRIORITY_TICKETS,

        COUNT_IF(PRIORITY = 'medium') AS MEDIUM_PRIORITY_TICKETS,

        COUNT_IF(PRIORITY = 'low') AS LOW_PRIORITY_TICKETS

    FROM tickets

    GROUP BY
        TICKET_TYPE,
        PRIORITY,
        QUEUE

)

SELECT *
FROM analytics
ORDER BY TOTAL_TICKETS DESC