## Methane HotSpot Dataset
Methane HotSpot Dataset is a large-scale dataset of methane plume segmentation mask for over 1200 AVIRIS-NG flight lines from 2015-2022. It contains over 4000 methane plume sites. 

[Dataset size is over 10 Tera Bytes. We created a sftp server to download the data, please follow in steps:]()


### Steps to download:
1. Open terminal on your linux machine or cmd windows system
2.  ```
    cd /location/where/to/download
    ```
3. Connect to SFTP server using the following command
    ```
    sftp -P 2022 anonymous@brain.ece.ucsb.edu
    ```
4. The data us available under "public" directory
    ```
    ls public
    ```
5. To download all data 
