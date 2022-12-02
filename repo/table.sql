CREATE DATABASE IF NOT EXISTS pacmann;

CREATE TABLE IF NOT EXISTS pacmann.cars (
    id VARCHAR(50),
    license_number VARCHAR(10),
    prediction VARCHAR(10),
    check_in_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_out_at TIMESTAMP
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
