# COMMON_LABEL_LIST = [
#         'text',
#         'receipt_id',
#         'date',
#         'time',
#         'product_name',
#         'product_quantity',
#         'product_total_money',
#         'total_money'
# ]

all_label_list = {
        '711': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'receipt_id',
                'time', 'total_discount_money', 'total_money', 'total_quantity'],
        'aeon_citimart': ['text', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_id', 'product_name', 'product_quantity',
                        'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money'],
        'aeon_combined': ['text', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                        'receipt_id', 'staff', 'time', 'total_money', 'total_quantity'],
        'BHD': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'time', 'total_money'],
        'bhx': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time', 'total_discount_money', 'total_original_money'],
        'bhx_2024': ['text', 'date', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'pos_id',
                'receipt_id', 'time', 'total_money'],
        'bigc_old': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_original_price', 'product_quantity', 'product_total_original_money',
                'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money', 'total_original_money', 'total_quantity'],
        'bonchon': ['text', 'date', 'mart_name', 'product_discount_money', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time', 'total_money', 'total_original_money'],
        'brg': ['text', 'date', 'mart_name', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'total_money'],
        'bsmart': ['text', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money'],
        'cheers': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money', 'total_quantity'],
        'circlek': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_discount_money', 'total_money', 'total_original_money', 'total_quantity'],
        'coopfood': ['text', 'barcode', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_original_price', 'product_quantity',
                                        'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money',
                                        'total_quantity'],
        'coopmart': ['text', 'barcode', 'date', 'mart_address', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_original_price', 'product_quantity',
                'product_total_money', 'product_unit_price', 'receipt_id', 'receipt_tax_number', 'staff', 'time', 'total_money', 'total_quantity'],
        'dmx': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'time', 'total_money'],
        'don_chicken': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'receipt_id', 'second_product_name',
                'staff', 'time', 'total_money', 'total_original_money'],
        'emart': ['text', 'barcode', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_id', 'product_name', 'product_original_price', 'product_quantity',
                'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money',
                'total_original_money', 'total_quantity'],
        'familymart': ['text', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_name', 'product_quantity', 'product_total_money',
                'product_unit_price', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money', 'total_original_money', 'total_quantity'],
        'fujimart': ['text', 'date', 'mart_name', 'product_id', 'product_name', 'product_original_price', 'product_quantity', 'product_total_money',
                'product_unit_price', 'receipt_id', 'staff', 'total_money', 'total_quantity'],
        'galaxy_cinema': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money'],
        'globalx': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'receipt_id', 'staff', 'time', 'total_money',
                'total_original_money'],
        'gs25': ['text', 'date', 'pos_id', 'product_discount_money', 'product_id', 'product_name', 'product_original_price', 'product_quantity',
                'product_total_money', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money', 'total_original_money',
                'total_quantity'],
        'guardian': ['text', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time', 'total_money'],
        'hc': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'time',
                'total_discount_money', 'total_money', 'total_original_money'],
        'heineken': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_discount_money', 'total_original_money', 'total_money'],
        'heineken_2024': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_original_money', 'total_money'],
        'kfc': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_total_money', 'receipt_id', 'staff', 'time', 'total_money', 'total_original_money'],
        'kingfood': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'time',
                'total_money', 'total_quantity'],
        'lamthao': ['text', 'date', 'mart_name', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money', 'total_quantity'],
        'lotte': ['text', 'date', 'mart_name', 'pos_id','product_discount_money', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time','total_discount_money', 'total_money', 'total_quantity'],
        'lotte-drop-0.4': ['text', 'date', 'mart_name', 'pos_id','product_discount_money', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time','total_discount_money', 'total_money', 'total_quantity'],
        'lotte_cinema': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money'],
        'lotteria': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money'],
        'mega_2022': ['text', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_id', 'product_name', 'product_quantity', 'product_total_money',
                'product_unit_price', 'receipt_id', 'staff', 'time', 'total_money', 'total_quantity', 'store_id'],
        'ministop': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'pos_id', 'time',
                'total_discount_money', 'total_money', 'total_original_money'],
        'newbigc_go_top': ['text', 'barcode', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_id', 'product_name',
                        'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time', 'total_money',
                        'total_quantity'],
        'new_gs25': ['text', 'date', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time',
                        'total_discount_money', 'total_money'],
        'nguyenkim': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'store_id', 'time', 'total_money'],
        'nova': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'time', 'total_money'],
        'nuty': ['text', 'date', 'mart_name', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'time', 'total_money', 'total_quantity'],
        'okono': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'time', 'total_discount_money', 'total_money', 'total_original_money'],
        'pepper_lunch': ['text', 'date', 'mart_name', 'pos_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id', 'staff',
                'time', 'total_money'],
        'pizza_company': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'receipt_id', 'staff', 'time', 'total_money',
                'total_original_money'],
        'satra': ['text', 'barcode', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'staff', 'time', 'total_money'],
        'tgs': ['text', 'date', 'mart_name', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'staff', 'time', 'total_money'],
        'thegioiskinfood': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                        'time', 'total_discount_money', 'total_money', 'total_original_money', 'total_quantity'],
        'winmart_combined': ['text', 'date', 'mart_name', 'pos_id', 'product_discount_money', 'product_id', 'product_name', 'product_quantity',
                        'product_total_money', 'product_unit_price', 'receipt_id', 'staff', 'time', 'total_discount_money', 'total_money'],
        'bitis': ['text', 'mart_name', 'receipt_id', 'date', 'time', 'product_name', 'product_unit_price', 'product_quantity', 'product_total_money',
                        'total_quantity', 'total_money', 'product_discount_retail_money', 'product_discount_wholesale_money', 'address'],
        'ushimania-yoshinoya': ['text', 'mart_name', 'receipt_id', 'date', 'staff', 'time', 'product_name', 'product_quantity', 'product_total_money', 'total_money', 'total_original_money'],
        'sayaka': ['text', 'mart_name', 'receipt_id', 'date', 'time', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'total_discount_money',
                'total_money', 'total_original_money'],
        'temp': ['text', 'total_money', 'product_name', 'product_total_money', 'product_quantity', 'staff', 'time', 'date', 'receipt_id', 'mart_name'],
#------------------------------------------------------------------------------ VAT--------------------------------------------------------------------
        'vat_caophong': ['text', 'date', 'mart_name', 'pos_id', 'product_id', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price',
                'receipt_id', 'total_money'],
        'vat_nguyenkim': ['text', 'date', 'mart_name', 'product_name', 'product_quantity', 'product_total_money', 'product_unit_price', 'receipt_id',
                'total_money', 'total_original_money'],
                        }


if __name__ == '__main__':
    common_labels = set(all_label_list['circlek']) & set(all_label_list['bitis']) & set(all_label_list['711']) & set(all_label_list['emart']) & set(all_label_list['new_gs25'])
    print(common_labels)

    for mart_name in ['winmart_combined', 'newbigc_go_top', 'coopmart', 'emart', 'gs25']:
        label_list = all_label_list[mart_name]
        print(f'{mart_name}: {len(label_list)}')