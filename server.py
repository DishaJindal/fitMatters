from flask import Flask
from flask import request
from flask import jsonify
import json
import datetime
from datetime import datetime, timedelta
from werkzeug import secure_filename
import os
import subprocess
from flask_cors import CORS


now = datetime.today()

print(now)

epoch = datetime.utcfromtimestamp(0)

app = Flask(__name__)
CORS(app)

sizes = ["S", "M", "L"]

class Listing:
    def __init__(self, brand, size, category):
        self.brand = brand
        self.size = size
        self.category = category
        # self.url = url

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)



class listingResponse:
    def __init__(self,name,size):
        self.url = "https://i2.wp.com/paristylebd.com/wp-content/uploads/2016/01/black-formal-2.jpg"
        self.name = name
        self.sizes = sizes
        self.activeSize = size

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


class Response:
    def __init__(self, listingResponse):
        self.listings = listingResponse

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


class SizeMeasurements:
    def __init__(self, width, length):
        self.width = width
        self.length = length


class Order:
    def __init__(self,id, item, rating, isreturn, orderPlaced):
        self.id = id
        self.item = item
        self.rating = rating
        self.isreturn = isreturn
        self.orderPlaced = orderPlaced


userSize = {}
userSize['Order_History'] = ""
userSize['Photo'] = ""
userSize['Both'] = ""

listings = [Listing("Reebok","S","shirt"),Listing("Reebok","M","shirt"),Listing("Reebok","L","shirt"),Listing("Nike","S","shirt"),Listing("Nike","M","shirt"),Listing("Nike","L","shirt"),Listing("Puma","S","shirt"),Listing("Puma","M","shirt"),Listing("Puma","L","shirt"),Listing("Tommy","S","shirt"),Listing("Tommy","M","shirt"),Listing("Tommy","L","shirt")]


measurement2size = {}
measurement2size["48_69"] = "Reebok_S"
measurement2size["50_69"] = "Reebok_M"
measurement2size["51_71"] = "Reebok_L"
measurement2size["42_67"] = "Nike_S"
measurement2size["43_69"] = "Nike_M"
measurement2size["45_71"] = "Nike_L"
measurement2size["38_69"] = "Puma_S"
measurement2size["40_72"] = "Puma_M"
measurement2size["43_74"] = "Puma_L"
measurement2size["40_69"] = "Tommy_S"
measurement2size["43_70"] = "Tommy_M"
measurement2size["45_72"] = "Tommy_L"

size2measurements = {}
for key in measurement2size:
    size2measurements[measurement2size[key]] = key

orders = [Order(1,"Reebok_S",2,False, (now - timedelta(days=60)))  , Order(2,"Reebok_M",5,False, (now - timedelta(days=30))) ,Order(3,"Nike_L",4,False,(now - timedelta(days=40))) ,Order(4,"Reebok_L",3,False, (now - timedelta(days=10))) ,Order(5,"Nike_S",1,False, (now - timedelta(days=8))) ,Order(6,"Nike_M",4,False, (now - timedelta(days=3)))]

order2Score = {}
orderId2item = {}
bestScore = -100
bestOrder = 0
for order in orders:
    score = order.rating * 0.8
    months = (now - order.orderPlaced).days / float(30)
    print(months)
    score = score + (0.2 / months)
    order2Score[order.id] = score
    if score > bestScore:
        bestOrder = order.id
    orderId2item[order.id] = order.item

print(order2Score)
print(orderId2item[bestOrder])

avgWidth = 0
avgLength = 0
scoreSum = 0
for orderId in order2Score:
    scoreSum = scoreSum + order2Score[orderId]
    measure = size2measurements[orderId2item[orderId]]
    avgWidth = avgWidth +  int(measure.split("_")[0])*order2Score[orderId]
    avgLength = avgLength +  int(measure.split("_")[1])*order2Score[orderId]

print(avgWidth/scoreSum)
print(avgLength/scoreSum)

userSize['Order_History'] = str(int(avgWidth/scoreSum)) + '_' + str(int(avgLength/scoreSum))

@app.route("/")
def hello():
    return "Hello World!"


UPLOAD_FOLDER = '/Users/sagar.sahni/Desktop'
# UPLOAD_FOLDER = '/Users/abhishek.krishan/scripts/hackday'



@app.route('/upload', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['image']:
        img = request.files['image']
        print(img.filename)
        img_name = secure_filename(img.filename)
        saved_path = os.path.join(UPLOAD_FOLDER, img_name)
        img.save(saved_path)
        completed = subprocess.run(['python3.6', 'object_size.py','-i',saved_path,'-w',"8.56"])
        file = open("hw.txt","r")
        data = file.readlines()
        w = int(float(data[0]))
        l = int(float(data[1]))
        userSize['Photo'] = str(w) + '_' + str(l)
        updateBothSize()
        #subprocess.run(["python3.6 object_size.py -i ", saved_path + " -w 3.37"])
        return saved_path
        # return send_from_directory(UPLOAD_FOLDER,img_name, as_attachment=True)
    else:
        return "Where is the image?"



@app.route("/updateSize")
def updateSize():
    args = request.args
    width = args['width']
    length = args['length']
    userSize['Photo'] = str(width) + '_' + str(length)
    updateBothSize()
    return "DONE"

def updateBothSize():
    width = int( int(userSize['Photo'].split("_")[0]) * 0.3  +  int(userSize['Order_History'].split("_")[0]) * 0.7 ) 
    length = int( int(userSize['Photo'].split("_")[1]) * 0.3  +  int(userSize['Order_History'].split("_")[1]) * 0.7 ) 
    userSize['Both'] = str(width) + '_' + str(length)


def getMySizeFromMeasurements(size):

    mywidth = int(size.split("_")[0])
    mylength = int(size.split("_")[1])
    brandSizeToDiff = {}

    for key in measurement2size:
        print(key)

        width = int(key.split("_")[0])
        length = int(key.split("_")[1])

        widthDiff = mywidth - width
        lengthDiff = mylength - length

        value = measurement2size[key]

        brand = value.split("_")[0]
        s = value.split("_")[1]

        print(value)

        if brandSizeToDiff.get(brand) is None: 
            brandSizeToDiff[brand] = s + "_" + str(widthDiff) + '_' + str(lengthDiff)
        else:
            prevValue = brandSizeToDiff[brand]

            prevsize = prevValue.split("_")[0]
            prevWidthDiff = int(prevValue.split("_")[1])
            prevLengthDiff = int(prevValue.split("_")[2])

            diff = widthDiff + lengthDiff
            prevDiff = prevWidthDiff + prevLengthDiff
            if ( abs(prevDiff) - abs(diff) > 0 and prevDiff * diff > 0 ) or (prevDiff * diff < 0 and diff > 0) or ( prevDiff == 0 and ((prevWidthDiff < 0 and widthDiff > 0)  or (prevWidthDiff> 0 and widthDiff < prevWidthDiff))) or (diff == 0 and widthDiff == 0 and lengthDiff == 0 ):
                brandSizeToDiff[brand] = s + "_" + str(widthDiff) + '_' + str(lengthDiff)

        print(brandSizeToDiff)

      
    print (brandSizeToDiff)

    lists = []
    for key in brandSizeToDiff:
        lists.append(listingResponse(key, brandSizeToDiff[key].split("_")[0]))
        # lists.append(Listing(key,brandSizeToDiff[key].split("_")[0],"tshirt"))


    return lists



@app.route("/userSizes")
def userSizes():
    return jsonify(userSize) 


@app.route("/catalogue/products")
def list():
    args = request.args
    print (args)
    size = args.get('size')
    sizeType = args.get('pref')
    brand = args.get('brand')
    results = []
    results1 = []
    print("sagar")
    print(size)
    print(brand)

    newL = []
    for l in listings:
        newL.append(listingResponse(l.brand,None))

    if brand is None and sizeType is None and size is None:
        return Response(newL).toJSON()


    if sizeType is not None:
        print(sizeType)
        measurements = userSize[sizeType]
        print(measurements)
        brandWithSizes = getMySizeFromMeasurements(measurements)

        print(brandWithSizes)
        if brand is not None:
            for b in brandWithSizes:
                if b.name.split("_")[0] == brand:
                    results1.append(b)
        else:
            results1 = brandWithSizes
        print(results1)
    else:   
        if brand is not None: 
            for listing in newL:
                if listing.name.split("_")[0] == brand:
                    results.append(listing)
        else:
            results = newL
        
        if size is not None:
            for result in results:
                print(result.activeSize)
                if result.activeSize == size:
                    results1.append(result)
        else: 
            results1 = results

    return Response(results1).toJSON()



if __name__ == '__main__':
    app.run(port=int("8080"),debug=True)
