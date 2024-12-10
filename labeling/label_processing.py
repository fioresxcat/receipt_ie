from utils.utils import *

def fake_label_guardian():
    json_dir = 'raw_data/adtima_data/guardian/text_jsons'
    im_dir = 'raw_data/adtima_data/guardian/warp_images'

    jpaths = sorted(list(Path(json_dir).glob('*.json')))
    for jp in jpaths:
        with open(jp) as f:
            data = json.load(f)
        all_labels = [shape['label'] for shape in data['shapes']]
        # if not list(set(all_labels)) == ['text']:
        #     continue
        if 'mart_name_block' in all_labels or 'pos_id' in all_labels:
            continue
        bb2label, bb2text, rbbs, bb2idx_original = sort_json(data)
        shapes = data['shapes']
        for row_idx, row in enumerate(rbbs):
            first_bb = row[0]
            first_bb_text = bb2text[first_bb].lower()
            if len(row) in [2,4] and first_bb_text.startswith('vat'):
                is_product_info_row = True
            else:
                is_product_info_row = False
            
            if len(row) > 1 and len(first_bb_text) > 10 and all(c.isdigit() for c in first_bb_text):
                is_product_name_row = True
            else:
                is_product_name_row = False

            if is_product_name_row and is_product_info_row:
                continue
            
            row_text = ' '.join([bb2text[bb] for bb in row])
            uncase_row_text = unidecode.unidecode(row_text).lower().replace(' ', '')
            for bb_idx, bb in enumerate(row):
                label = bb2label[bb]
                if label != 'text':
                    continue
                text = bb2text[bb]
                uncase_text = unidecode.unidecode(text).lower()
                orig_idx = bb2idx_original[bb]
                new_label = 'text'
                if is_product_name_row:  # this is product id
                    if bb_idx == 0:
                        new_label = 'product_id'
                    else:
                        new_label = 'product_name'

                elif is_product_info_row:
                    if bb_idx == 1 and len(text) < 2:
                        new_label = 'product_quantity'
                    elif bb_idx == 2 and len(row) == 4:
                        new_label = 'product_unit_price'
                    elif bb_idx == 3 and len(row) == 4:
                        new_label = 'product_total_money'
                    elif bb_idx == 2 and len(row) == 3:
                        new_label = 'product_total_money'
                
                elif 'tongtienduocgiam' in uncase_row_text and bb_idx == len(row) - 1:
                    new_label = 'total_discount_money'
                
                elif 'tongtienduocgiam' in uncase_row_text and bb_idx == len(row) - 1:
                    new_label = 'total_discount_money'
                
                elif 'canthanhtoan' in uncase_row_text and bb_idx == len(row) - 1:
                    new_label = 'total_money'
                
                elif 'tongsoluonghang' in uncase_row_text and bb_idx == len(row) - 1:
                    new_label = 'total_quantity'
                
                shapes[orig_idx]['label'] = new_label
            
        data['shapes'] = shapes

        with open(jp, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

        print(f'done {jp}')


def get_word_label_in_block():
    json_dir = 'raw_data/adtima_data/watsons/text_jsons'
    jpaths = sorted(list(Path(json_dir).glob('*.json')))
    for jp in jpaths:
        with open(jp) as f:
            data = json.load(f)

        block_shapes = []
        for shape in data['shapes']:
            if len(shape['points']) == 2 and '_block' in shape['label']:
                block_shapes.append(shape)
        
        for block in block_shapes:
            new_label = block['label'].replace('_block', '')
            for shape_idx, shape in enumerate(data['shapes']):
                if shape in block_shapes or shape['label'] != 'text':
                    continue
                r1, r2, iou = iou_poly(shape['points'], block['points'])
                if r1 > 0.7:
                    data['shapes'][shape_idx]['label'] = new_label
        
        with open(jp, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f'done {jp}')


if __name__ == '__main__':
    pass
    # fake_label_guardian()
    get_word_label_in_block()
