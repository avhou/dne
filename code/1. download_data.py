import requests
import pandas as pd
import io
from configparser import ConfigParser
import os
from tqdm import tqdm
import time
from typing import Callable, Any

# Read the filepath from the config ini file
config = ConfigParser()
config.read("config.ini")
target_file_path = config.get("Data", "SolarFilePath")
retry_attempts = config.get("Data", "RetryAttempts")

# the elia hashes linked to each dataset
elia_dataset_hashes = [
    "a638db02316931af6562a9eef9a89073",
    "e457d1a2d7a349d8c31a767fa4e50947",
    "d602845ff8d0e4d1db7faa3cc1d945f0",
    "78db39a629611556d71a91d8bd77cf37",
    "edaa12884f395e27266d50f20abe0837",
    "d11d5e50d8e4921ac4a5e7d416ac3f40",
    "65155211eee7406e65077b6fb87d6642",
    "5f50b6c5e8f298e496001fb7b9b49824",
    "b6ff3b634779314df86f5848971a25fd",
    "53f068ee19526bb8d20608d06d1daacd",
    "523839c49dbf6b37d0e1141749800b46",
    "14cadde78bd2b9c4203f6dbaafcfac94",
    "2b768518f89a974c4b786d5f3b1f16c3",
    "67ab9f99eb2072b8c671b68da6a8fd87",
    "8860d4398c9beb5cd56ea2016b90a537",
    "71ef747e18564dba242da82a46a54898",
    "b33c0c7d3b9c4879714af736bc8910ea",
    "f7360f2cf374c63e2c63f6ee4ad53d18",
    "40b2761fd96c6d70f429507c558664e4",
    "6291a7bfc279fd785b8e3384eb4b097b",
    "0707a72d4219afcbd1e156cd78ee62e0",
    "c1d7bf7b7a8e24ad5d647c26ddaa08c7",
    "2b3e69448eddd3c3e7b812eebf46cbfc",
    "eb64f1c2a265ae58e4ec66fbfd5fc055",
    "45a6024238fd64e9538fd4c7e47b0ffc",
    "d40371d5017ba018f1c40e8f322a669a",
    "3a8abeead42ccd286a7d0cbd4af82313",
    "3ab14b4d86ea7e604cd93de04c71ef51",
    "c75264a105abfe94f6bfeca75e5c5cfc",
    "1e84ea817466f519520dbf277403a056",
    "4d07a3bdc1f6c260f91eff5a8c4732c0",
    "d232553855d058ce3dc6ea0a078472a1",
    "fc81369ba22ecb70ddaf5da1b9ca5322",
    "76bc3a375f8e5795348c2e28143619d9",
    "197010c3979cec07a5f3c577bedb527d",
    "bde49e4082500b233767ac0393f52c97",
    "030c69314a5026a51c56546f16b3d52c",
    "e9d3c4f3be58e8eb9a2fe5ef80b5dbcd",
    "bfc4ee5021155fcdb3f00ba94f787ff5",
    "1e206205087c8a73eb89aea29d5fabd9",
    "6f3600adb9ad41515d677f040115745c",
    "684b2b6d92de5c0fa5f2d605fd16308d",
    "97baae9a0464e15f380940bbfc9b5078",
    "dd00f05df01319dd2768e8101966d569",
    "be0c84ff5b19b7be3f3c3bcf63481301",
    "8bba7436b8e1ce3bf9b36693fe576274",
    "3dd5f9e9f41f3fbaae4d982e5708ba04",
    "c8898f19b418e301cebe9fa799d6be4a",
    "ca67bdac3e5296c9aca14c7757a45a43",
    "bc407ceaf8eb27d5253807a9ebda0414",
    "19292e615ff604487bbe1640090be1cc",
    "5f5029abe1b4cbc48fdee78126b6f437",
    "9dd94ddff48ae07e14aec65b7ae84a32",
    "0960d7e21fb4988364941e32701e571d",
    "75068ba7863956b814a3948f1017d0c4",
    "c2ede9adb95165c880813f5891c578cb",
    "d7d6bcc7c5ea1851a35bb78ab1403fd8",
    "2f34af5247b464f4b85b2be46024d9d1",
    "d16d81a8cdccc1b9f179cf8058053c52",
    "748c92a1d4464519edcb4b56646b5407",
    "d97683d280d36a8f907f655c8ae74044",
    "ba0f2e0bdaa41723c20bcd9e8ce34685",
    "5ce97e5ef051c8c98e68c92d3332cb57",
    "bd8625650d5d8201bac0d438b6b8bf4f",
    "6d045c08fe85f809944922562416c01a",
    "27328ef4198621c47c9eed9d679e106e",
    "8c12b0d5d2c29f9c75a842dfa46dab38",
    "e66b5a3c6799dcc5dd4039a58085fb38",
    "af77dfd404964d21108cf737bd810e11",
    "2c93879227a2e2e70ce85ae352f8f46c",
    "a9dbcc79de4333ae553783d094cf656c",
    "effdc6a5d743a9db1cd347a2ac8d6b80",
    "95798f037b7bacec7599a19423a6bb64",
    "74620215d33d2f9ad06a873cdf9e8acc",
    "87dddbc407376ab2a74d919deb8a64ec",
    "9fe5b0e5e0c862102d23c193d848d031",
    "bc53caa65f8023df6c24c6481bb49196",
    "e3084455355bf91c7240d9fdb4d6774d",
    "0ba0e8d5409b2993f9ed1e11567be7c3",
    "a0412a4f0a7daea31c561a20fad808b8",
    "77fce67ff1ced73f063cdd19c439b627",
    "a10ba827ae758dee0e50d44366fd3d6d",
    "472471538784273d4b38b8713a92089d",
    "1a4a2c4586c0aaea89f273c2ef7b2472",
    "faae77443f7a8406deaf55c3795ace10",
    "692664770412ddd705b7f413cee7d6fb",
    "1b34d20d5f729538f1400e856387b228",
    "40c6992ea910d43026a8f13fd0d7bfd6",
    "e289ad98e91a5c5e09b44d6006fba214",
    "b630370a2d17f69ca45d015cd874da82",
    "06563fa1bfd9cc3baa74dcf102b25f95",
    "bbfb6d5a145f45ac380d5c56707a3b3f",
    "e10986a4d8d7c27e4a9031d551cb15f3",
    "aafb48343bbcbc586f011c8f87a52d7f",
    "42aef0714dca75b922449334f8354e29",
    "d835981d77534f5f20446d4648ad7e5b",
    "cf2dfa7ef96d78d1f3ec76316e1481d0",
    "342aa26fa2cc7e365c1b819a4e86b6f1",
    "a00f05a660c5d5d561663fe44e44cdf5",
    "0fa56213ed5deea97598ccd1f1d3db87",
    "0c32ac23d148fcf8f103092f860ea484",
    "ca82c7bf0a12e1bd7623c916688916ff",
    "ea6b0fa25e3cb324da39980f6290f837",
    "a3da3f6c7a5a3a1b5123b65948792dbb",
    "d4f4c40bd169a262676284f5da7a191a",
    "0e347c50a0d60b8f46cb8c37cd3b5088",
    "a5667679a016ee66ef9bc45c1a173b3b",
    "0de26c1011bb34bba047de0a023c9e86",
    "8aa90565fb34d8b0aeab74ab74ffdf4a",
    "5e79a20254a28ffdb604f6cba5216c75",
    "9a4bface9098ff4cd4b35cca641c0042",
    "0933d4dd2ad52707372be6b67b620890",
    "db5be383ed95aea37a04f941990f2c45",
    "3e253f251ad932b01d5cb5e83eb56183",
    "248f830b158797f9c038ea35ea266b89",
    "d802637b72ebddd27cf9b28f9f198763",
    "a5f3f63c2f6e1d6d4605650633b9ce8a",
    "62783f5d69cb5d3cb22b077c1d2b8777",
    "a5091aedd97daf7c5a5297fa5d73a469",
    "b9365bb9132d1eb8770275a763fc5d69",
    "9209b4ae424c71673a607557c64f7eb8",
    "e50e258ecc9eaa88d6c653c3a24cb319",
    "36098c0d1c3de71d8727b1543bace633",
    "b56a5394775a9f077c12cb3e770d913c",
    "95eafe7994082a98ef565bc49b0bd0bc",
    "d1cdbd3c8324f3afbc5b420ce471feff",
    "8bab215794a736f6f876eac0eeb23923",
    "1b70ac0a972fddf1e1a33e6ed9df49c7",
    "4c4518d4c20175eb4082a6b33eb4badb",
    "01a314c1b9df8a1e936aee8eb0691f89",
    "f5616c23f020b3f984cbcd5f9a053ecc",
    "abb14d21829ba186172f6e1d0b64b55b",
    "58d98f8209a2b3224b4cf173fb43579c",
    "fb4d860c1938626a4cb9b8f66c6eb6f2",
    "904837c9cf9bc52a96ce17d051fddec9",
    "e16c8047fba82583090b96bf5e42b694",
    "411d690288852abddc39378b2989bfda",
    "fb5978407d362c06c4a1c8ab72f464f1",
    "6d6bbe0ff14c220c87649b72d24ad927",
    "4ae43ffefdaf3c2343be08c2c231978f",
    "b0ec3d3ab3b33bd69418a7bd9e4d8f9e",
    "fb784db76f57bc935639ba5340089d77",
    "c33f6b520ad3c202db9ed1b825ac8b3d",
    "efe846805e5476fec607bbf54e45dc44",
    "f76b3457c69f53525d8080927e20d41d",
]

# Downloads solar datasets from the Elia API and saves them as a CSV file.
# This function reads the file path from the config.ini file, downloads multiple solar datasets
# from the Elia API, and saves them as a single CSV file specified in the config.ini file.


def download_elia_dataset(elia_dataset_hash: str, retry_attempts: int = 5):
    download_successful = False
    current_retry = 1
    while not download_successful:
        print(f"Downloading dataset {elia_dataset_hash}... Attempt {current_retry}/{retry_attempts}.")
        url = f"https://griddata.elia.be/eliabecontrols.prod/interface/fdn/download/solarweekly/{elia_dataset_hash}?dtFrom=2012-03-01&dtTo=2024-03-30&sourceID=1&forecast=solar"
        http_response = requests.get(url, timeout=1000)
        download_successful = http_response.status_code == 200
        current_retry += 1
        time.sleep(2 ^ retry_attempts)
        if current_retry == retry_attempts:
            print(f"Critical failure after {retry_attempts} attempts.")
            break
    return http_response.content


def save_elia_dataset(elia_dataset_hash: str, file_path: str, retry_attempts: int = 5):
    if not os.path.exists(file_path):
        http_content = download_elia_dataset(elia_dataset_hash, retry_attempts=retry_attempts)
        with open(file_path, "wb") as file:
            file.write(http_content)


def concat_elia_datasets(file_paths: list[str]):
    full_df = pd.DataFrame()
    for file_path in file_paths:
        df = pd.read_excel(file_path, skiprows=3)  # Specify the engine parameter as 'openpyxl'
        full_df = pd.concat([full_df, df], ignore_index=True)

    return full_df


file_paths = []
for elia_dataset_hash in tqdm(elia_dataset_hashes, desc="Downloading Elia solar datasets..."):
    file_path = f"solar_data_{elia_dataset_hash}.xls"
    save_elia_dataset(elia_dataset_hash, file_path)
    file_paths.append(file_path)

full_df = concat_elia_datasets(file_paths)
full_df.sort_values(by="DateTime", inplace=True)
full_df.to_csv(target_file_path, index=False)
[os.remove(file_path) for file_path in file_paths]
