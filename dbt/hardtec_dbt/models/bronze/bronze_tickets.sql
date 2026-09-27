{{ config(materialized='table') }}

SELECT
    SUBJECT,
    BODY,
    ANSWER,
    TYPE,
    QUEUE,
    PRIORITY,
    LANGUAGE,
    TAG_1,
    TAG_2,
    TAG_3,
    TAG_4,
    TAG_5,
    TAG_6,
    TAG_7,
    TAG_8
FROM HARDTEC_DB.PUBLIC.TICKETS_RAW